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

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file operations including text extraction and file moving."""

    def __init__(self, source_paths: List[str], ollama_endpoint: str = "http://localhost:11434",
                 ocr_model: str = "deepseek-ocr:3b", ocr_timeout: int = 60):
        """Initialize file handler.

        Args:
            source_paths: List of source directories to read files from
            ollama_endpoint: Ollama API endpoint for OCR
            ocr_model: OCR model name (default: deepseek-ocr:3b)
            ocr_timeout: Timeout for OCR operations in seconds
        """
        self.source_paths = [Path(path) for path in source_paths]
        self.ollama_endpoint = ollama_endpoint
        self.ocr_model = ocr_model
        self.ocr_timeout = ocr_timeout

        # Initialize Ollama client for OCR
        self.ocr_client = ollama.Client(host=ollama_endpoint, timeout=ocr_timeout)
        self.ocr_available = self._check_ocr_availability()

        # Ensure source directories exist
        for source_path in self.source_paths:
            source_path.mkdir(parents=True, exist_ok=True)

    def _check_ocr_availability(self) -> bool:
        """Check if DeepSeek-OCR is available and accessible."""
        try:
            # Try a simple test call to check availability
            response = self.ocr_client.generate(
                model=self.ocr_model,
                prompt="test",
                options={'num_predict': 1}  # Minimal response
            )
            return True
        except Exception:
            logger.warning(f"DeepSeek-OCR not available at {self.ollama_endpoint}")
            return False
    
    def get_files(self, extensions: Optional[List[str]] = None, recursive: bool = True) -> List[Path]:
        """Get list of files from source directories.

        Args:
            extensions: List of file extensions to filter (e.g., ['.pdf', '.docx'])
                       If None, returns all files
            recursive: Whether to search recursively

        Returns:
            List of file paths
        """
        files = []

        for source_path in self.source_paths:
            if not source_path.exists():
                logger.warning(f"Source path does not exist: {source_path}")
                continue

            pattern = "**/*" if recursive else "*"

            for file_path in source_path.glob(pattern):
                if file_path.is_file():
                    # Filter by extension if specified
                    if extensions:
                        if file_path.suffix.lower() in extensions:
                            files.append(file_path)
                    else:
                        files.append(file_path)
        
        return files
    
    def extract_text(self, file_path: Path) -> tuple[Optional[str], float]:
        """Extract text content from a file.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (extracted text content or None if extraction fails, OCR duration in seconds)
        """
        # Ensure file_path is a Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
        """Extract text content from a file.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content or None if extraction fails
        """
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
                            return (ocr_text, ocr_duration)
                        else:
                            return (text, 0.0)
                    else:
                        logger.warning(f"DeepSeek-OCR not available, skipping OCR for {file_path}")
                        return (text, 0.0)
                return (text, 0.0)
            elif suffix == '.docx':
                return (self._extract_from_docx(file_path), 0.0)
            elif suffix == '.doc':
                return (self._extract_from_doc(file_path), 0.0)
            elif suffix == '.txt':
                return (self._extract_from_txt(file_path), 0.0)
            elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp']:
                if self.ocr_available:
                    text, ocr_duration = self._extract_from_image(file_path)
                    if text:
                        logger.info(f"DeepSeek-OCR successfully extracted text from {file_path}")
                        return (text, ocr_duration)
                    else:
                        logger.warning(f"DeepSeek-OCR failed to extract text from {file_path}")
                        return (None, 0.0)
                else:
                    logger.warning(f"DeepSeek-OCR not available, cannot process image {file_path}")
                    return (None, 0.0)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return (None, 0.0)

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return (None, 0.0)
    
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
        doc = Document(file_path)
        text_parts = []
        for paragraph in doc.paragraphs:
            text_parts.append(paragraph.text)
        return '\n'.join(text_parts)

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
    
    def _extract_from_image(self, file_path: Path) -> tuple[str, float]:
        """Extract text from image using DeepSeek OCR."""
        try:
            image = Image.open(file_path)
            ocr_result = self._ocr_image_with_deepseek(image)
            if ocr_result:
                text, duration = ocr_result
                return (text, duration)
            else:
                return ("", 0.0)
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            return ("", 0.0)

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

            # Call DeepSeek-OCR with timing
            # Based on https://ollama.com/library/deepseek-ocr examples
            start_time = time.time()
            response = self.ocr_client.generate(
                model=self.ocr_model,
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
    

