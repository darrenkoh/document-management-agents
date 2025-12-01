"""File handling operations for reading files."""
import logging
import hashlib
import base64
import io
import subprocess
import time
from pathlib import Path
from typing import Optional, List
from pypdf import PdfReader
from docx import Document
from PIL import Image
import ollama

# Try to import pdf2image for PDF to image conversion
try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

from src.backend.utils.retry import retry_on_llm_failure, RetryError

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file operations including text extraction and file moving."""

    def __init__(self, source_paths: List[str], ollama_endpoint: str = "http://localhost:11434",
                 ocr_model: str = "deepseek-ocr:3b", ocr_timeout: int = 60,
                 max_retries: int = 3, retry_base_delay: float = 1.0):
        """Initialize file handler.

        Args:
            source_paths: List of source directories to read files from
            ollama_endpoint: Ollama API endpoint for OCR
            ocr_model: OCR model name (default: deepseek-ocr:3b)
            ocr_timeout: Timeout for OCR operations in seconds
            max_retries: Maximum number of retry attempts for failed API calls
            retry_base_delay: Base delay in seconds between retry attempts
        """
        self.source_paths = [Path(path) for path in source_paths]
        self.ollama_endpoint = ollama_endpoint
        self.ocr_model = ocr_model
        self.ocr_timeout = ocr_timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        # Initialize Ollama client for OCR
        self.ocr_client = ollama.Client(host=ollama_endpoint, timeout=ocr_timeout)
        self.ocr_available = self._check_ocr_availability()

        # Ensure source directories exist
        for source_path in self.source_paths:
            source_path.mkdir(parents=True, exist_ok=True)

    def _call_ocr_generate(self, prompt: str, images: Optional[List[str]] = None, options: Optional[dict] = None) -> dict:
        """Make OCR LLM generate call with retry logic."""
        @retry_on_llm_failure(max_retries=self.max_retries,
                             base_delay=self.retry_base_delay,
                             exceptions=(Exception,))  # OCR can fail for various reasons
        def _generate():
            generate_kwargs = {
                'model': self.ocr_model,
                'prompt': prompt,
                'options': options or {'num_predict': 1}
            }
            if images:
                generate_kwargs['images'] = images
            return self.ocr_client.generate(**generate_kwargs)

        return _generate()

    def _check_ocr_availability(self) -> bool:
        """Check if DeepSeek-OCR is available and accessible."""
        try:
            # Try a simple test call to check availability
            self._call_ocr_generate(
                prompt="test",
                options={'num_predict': 1}  # Minimal response
            )
            return True
        except RetryError as e:
            logger.warning(f"DeepSeek-OCR not available at {self.ollama_endpoint} after retries: {e.last_exception}")
            return False
        except Exception:
            logger.warning(f"DeepSeek-OCR not available at {self.ollama_endpoint}")
            return False
    
    def _is_included(self, file_path: Path, allowed_extensions: Optional[List[str]] = None) -> bool:
        """Check if a file path should be included for processing.

        Args:
            file_path: Path to the file to check
            allowed_extensions: List of allowed file extensions (lowercase)

        Returns:
            True if the file should be included, False otherwise
        """
        if not allowed_extensions:
            return True  # If no extensions specified, include all files

        return file_path.suffix.lower() in allowed_extensions
    
    def get_files(self, extensions: Optional[List[str]] = None, recursive: bool = True) -> List[Path]:
        """Get list of files from source directories.

        Args:
            extensions: List of file extensions to include (e.g., ['.pdf', '.docx'])
                       If None, returns all files
            recursive: Whether to search recursively

        Returns:
            List of file paths that match the inclusion criteria
        """
        files = []

        for source_path in self.source_paths:
            if not source_path.exists():
                logger.warning(f"Source path does not exist: {source_path}")
                continue

            pattern = "**/*" if recursive else "*"

            for file_path in source_path.glob(pattern):
                if file_path.is_file():
                    # Check if file should be included
                    if self._is_included(file_path, extensions):
                        files.append(file_path)
                    else:
                        logger.info(f"Skipping file with disallowed extension: {file_path}")
        
        return files
    
    def extract_text(self, file_path: Path) -> tuple[Optional[str], float, bool]:
        """Extract text content from a file.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (extracted text content or None if extraction fails, OCR duration in seconds, whether DeepSeek-OCR was used)
        """
        # Ensure file_path is a Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
        try:
            suffix = file_path.suffix.lower()

            if suffix == '.pdf':
                text = self._extract_from_pdf(file_path)
                # If PDF text extraction returns minimal content, try OCR
                if not text or len(text.strip()) < 50:  # Threshold for "empty" content
                    if self.ocr_available:
                        logger.info(f"PDF text extraction returned minimal content, trying DeepSeek-OCR for {file_path}")
                        ocr_result = self._extract_text_with_deepseek_ocr(file_path)
                        if ocr_result:
                            ocr_text, ocr_duration = ocr_result
                            logger.info(f"DeepSeek-OCR successfully extracted text from {file_path}")
                            return (ocr_text, ocr_duration, True)
                        else:
                            return (text, 0.0, False)
                    else:
                        logger.warning(f"DeepSeek-OCR not available, skipping OCR for {file_path}")
                        return (text, 0.0, False)
                return (text, 0.0, False)
            elif suffix == '.docx':
                text = self._extract_from_docx(file_path)
                # If DOCX text extraction returns minimal content, try OCR as fallback
                if not text or len(text.strip()) < 50:  # Threshold for "empty" content
                    if self.ocr_available:
                        logger.info(f"DOCX text extraction returned minimal content, trying DeepSeek-OCR for {file_path}")
                        ocr_result = self._extract_text_with_deepseek_ocr(file_path)
                        if ocr_result:
                            ocr_text, ocr_duration = ocr_result
                            logger.info(f"DeepSeek-OCR successfully extracted text from DOCX {file_path}")
                            return (ocr_text, ocr_duration, True)
                        else:
                            return (text, 0.0, False)
                    else:
                        logger.warning(f"DeepSeek-OCR not available, skipping OCR fallback for {file_path}")
                        return (text, 0.0, False)
                return (text, 0.0, False)
            elif suffix == '.doc':
                return (self._extract_from_doc(file_path), 0.0, False)
            elif suffix == '.txt':
                return (self._extract_from_txt(file_path), 0.0, False)
            elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp']:
                if self.ocr_available:
                    text, ocr_duration = self._extract_from_image(file_path)
                    if text is not None:
                        return (text, ocr_duration, True)
                    else:
                        logger.info(f"Skipping image {file_path} - no meaningful text content found")
                        return (None, ocr_duration, True)
                else:
                    logger.warning(f"DeepSeek-OCR not available, skipping image {file_path}")
                    return (None, 0.0, False)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return (None, 0.0, False)

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return (None, 0.0, False)
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text_parts = []
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
        return '\n'.join(text_parts)
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text_parts = []

            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            if paragraph.text.strip():
                                text_parts.append(paragraph.text)

            # Extract text from headers and footers
            for section in doc.sections:
                header = section.header
                if header:
                    for paragraph in header.paragraphs:
                        if paragraph.text.strip():
                            text_parts.append(paragraph.text)

                footer = section.footer
                if footer:
                    for paragraph in footer.paragraphs:
                        if paragraph.text.strip():
                            text_parts.append(paragraph.text)

            extracted_text = '\n'.join(text_parts)

            if not extracted_text.strip():
                logger.warning(f"No text content found in DOCX file {file_path}. File may be empty, corrupted, or contain only images.")
                return ""

            logger.info(f"Successfully extracted {len(extracted_text)} characters from DOCX file {file_path}")
            return extracted_text

        except Exception as e:
            logger.error(f"Failed to extract text from DOCX file {file_path}: {e}")
            return ""

    def _extract_from_doc(self, file_path: Path) -> str:
        """Extract text from DOC file using antiword."""
        try:
            # Try using antiword to extract text from .doc files
            result = subprocess.run(
                ['antiword', str(file_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout
            else:
                logger.warning(f"antiword failed for {file_path}: {result.stderr}")
                return ""
        except FileNotFoundError:
            logger.warning("antiword not installed. Install with: brew install antiword (macOS) or apt-get install antiword (Ubuntu)")
            return ""
        except subprocess.TimeoutExpired:
            logger.warning(f"antiword timed out for {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Failed to extract text from DOC file {file_path}: {e}")
            return ""
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_from_image(self, file_path: Path) -> tuple[Optional[str], float]:
        """Extract text from image using DeepSeek OCR.

        Returns:
            Tuple of (extracted text or None if no meaningful text found, OCR duration in seconds)
        """
        try:
            image = Image.open(file_path)
            ocr_result = self._ocr_image_with_deepseek(image)
            if ocr_result:
                text, duration = ocr_result
                # Check for meaningful content (not just whitespace or markdown boilerplate)
                stripped_text = text.strip() if text else ""
                if self._has_meaningful_content(stripped_text):
                    logger.info(f"DeepSeek-OCR successfully extracted {len(stripped_text)} characters of meaningful content from {file_path}")
                    return (text, duration)
                else:
                    logger.info(f"DeepSeek-OCR found no meaningful text content in image {file_path} (likely a photo, chart, or empty image). Extracted: '{stripped_text[:100]}{'...' if len(stripped_text) > 100 else ''}'")
                    return (None, duration)
            else:
                logger.warning(f"DeepSeek-OCR processing failed for {file_path}")
                return (None, 0.0)
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {e}")
            return (None, 0.0)

    def _has_meaningful_content(self, text: str) -> bool:
        """Check if text contains meaningful content (not just boilerplate, formatting, or OCR metadata).

        Args:
            text: The text to check

        Returns:
            True if text appears to contain meaningful content, False otherwise
        """
        if not text:
            return False

        # Remove common markdown boilerplate
        text = text.replace('```markdown', '').replace('```', '').strip()

        # Remove DeepSeek OCR metadata tokens
        text = text.replace('<|ref|>', '').replace('<|/ref|>', '').replace('<|det|>', '').replace('<|/det|>', '').strip()

        # Remove coordinate data (patterns like [[x,y,w,h]])
        import re
        text = re.sub(r'\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]', '', text).strip()

        # Check for minimum length (at least 10 characters for meaningful content)
        if len(text) < 10:
            return False

        # Check if it's just whitespace, punctuation, or common OCR artifacts
        if text.replace(' ', '').replace('\n', '').replace('\t', '').replace('-', '').replace('_', '').replace('=', '').replace('*', '').replace('#', '') == '':
            return False

        # Check for common "no content" responses
        no_content_indicators = [
            'no text found', 'no content', 'empty', 'blank',
            'no readable text', 'unable to extract', 'no data',
            'image', 'photo', 'picture', 'diagram', 'chart', 'graph'
        ]
        if any(indicator in text.lower() for indicator in no_content_indicators):
            return False

        # Check if text contains actual readable words (not just symbols)
        words = [word for word in text.split() if word.isalnum() and len(word) > 1]
        if len(words) < 2:  # Need at least 2 meaningful words
            return False

        return True

    def _pdf_to_images(self, file_path: Path) -> List[Image.Image]:
        """Convert PDF pages to PIL Images for OCR processing.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of PIL Images, one per page
        """
        try:
            if not HAS_PDF2IMAGE:
                logger.warning("pdf2image not available, cannot convert PDF to images for OCR")
                return []

            # Use pdf2image to convert PDF pages to images
            images = convert_from_path(file_path, dpi=200)  # 200 DPI for good OCR quality
            return images

        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []

    def _extract_text_with_deepseek_ocr(self, file_path: Path) -> Optional[str]:
        """Extract text from file using DeepSeek-OCR as fallback.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text or None if OCR fails
        """
        try:
            suffix = file_path.suffix.lower()

            if suffix == '.pdf':
                # Convert PDF to images first
                images = self._pdf_to_images(file_path)
                if not images:
                    # If we can't extract images, try to render PDF pages as images
                    # For now, return None - in production you might want to use pdf2image
                    logger.warning(f"Could not extract images from PDF: {file_path}")
                    return None

                # Process each image with DeepSeek-OCR
                all_text = []
                total_ocr_duration = 0.0
                for i, image in enumerate(images):
                    logger.info(f"Processing page {i+1}/{len(images)} with DeepSeek-OCR")
                    ocr_result = self._ocr_image_with_deepseek(image)
                    if ocr_result:
                        text, duration = ocr_result
                        all_text.append(text)
                        total_ocr_duration += duration
                    else:
                        logger.warning(f"Failed to extract text from page {i+1}")

                if all_text:
                    return ('\n'.join(all_text), total_ocr_duration)
                else:
                    return None

            elif suffix == '.docx':
                # For DOCX files, we need to convert to images first
                # This requires additional dependencies like docx2pdf and pdf2image
                logger.warning(f"DOCX OCR not yet implemented. Consider installing docx2pdf and pdf2image for DOCX OCR support.")
                return None

            elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp']:
                # Direct image OCR
                image = Image.open(file_path)
                ocr_result = self._ocr_image_with_deepseek(image)
                return ocr_result  # Already returns (text, duration) or None
            else:
                return None

        except Exception as e:
            logger.error(f"DeepSeek OCR failed for {file_path}: {e}")
            return None

    def _ocr_image_with_deepseek(self, image: Image.Image) -> Optional[tuple[str, float]]:
        """Perform OCR on a single image using DeepSeek-OCR.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (extracted text, duration in seconds) or None if failed
        """
        try:
            # Convert image to base64 string (not data URL)
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Call DeepSeek-OCR with timing and retry logic
            # Based on https://ollama.com/library/deepseek-ocr examples
            start_time = time.time()
            response = self._call_ocr_generate(
                prompt="<|grounding|>Convert the document to markdown.",
                images=[image_data],  # Base64 image data
                options={
                    'temperature': 0.0,  # Deterministic output for OCR
                    'num_predict': 2000,  # Increased for structured markdown output
                }
            )
            ocr_duration = time.time() - start_time

            extracted_text = response.get('response', '').strip()
            if extracted_text:
                logger.info(f"DeepSeek-OCR extracted text: {extracted_text[:100]}...")
                return (extracted_text, ocr_duration)
            else:
                logger.info(f"DeepSeek-OCR returned empty response. Full response: {response}")
                return None

        except RetryError as e:
            logger.error(f"DeepSeek OCR failed after retries: {e.last_exception}")
            return None
        except Exception as e:
            # Log connection issues but don't spam the logs for each image
            if "No route to host" in str(e) or "Connection refused" in str(e):
                logger.warning(f"DeepSeek OCR unavailable: Cannot connect to Ollama at {self.ollama_endpoint}. OCR fallback disabled.")
            else:
                logger.error(f"DeepSeek OCR failed: {e}")
            return None

    def generate_file_hash(self, file_path: Path) -> Optional[tuple[str, float]]:
        """Generate SHA-256 hash of the file content.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (SHA-256 hash as hexadecimal string, duration in seconds) or None if failed
        """
        import time
        try:
            start_time = time.time()

            # Read the entire file in binary mode
            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Generate SHA-256 hash
            hash_obj = hashlib.sha256(file_content)
            hash_value = hash_obj.hexdigest()

            duration = time.time() - start_time
            return (hash_value, duration)

        except Exception as e:
            logger.error(f"Error generating hash for {file_path}: {e}")
            return None
    

