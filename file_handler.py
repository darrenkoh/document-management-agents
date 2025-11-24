"""File handling operations for reading files."""
import logging
from pathlib import Path
from typing import Optional, List
from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file operations including text extraction and file moving."""
    
    def __init__(self, source_path: str):
        """Initialize file handler.
        
        Args:
            source_path: Source directory to read files from
        """
        self.source_path = Path(source_path)
        
        # Ensure source directory exists
        self.source_path.mkdir(parents=True, exist_ok=True)
    
    def get_files(self, extensions: Optional[List[str]] = None, recursive: bool = True) -> List[Path]:
        """Get list of files from source directory.
        
        Args:
            extensions: List of file extensions to filter (e.g., ['.pdf', '.docx'])
                       If None, returns all files
            recursive: Whether to search recursively
        
        Returns:
            List of file paths
        """
        files = []
        
        if not self.source_path.exists():
            logger.warning(f"Source path does not exist: {self.source_path}")
            return files
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in self.source_path.glob(pattern):
            if file_path.is_file():
                # Filter by extension if specified
                if extensions:
                    if file_path.suffix.lower() in extensions:
                        files.append(file_path)
                else:
                    files.append(file_path)
        
        return files
    
    def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text content from a file.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Extracted text content or None if extraction fails
        """
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.pdf':
                return self._extract_from_pdf(file_path)
            elif suffix in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            elif suffix == '.txt':
                return self._extract_from_txt(file_path)
            elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp']:
                return self._extract_from_image(file_path)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return None
        
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return None
    
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
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_from_image(self, file_path: Path) -> str:
        """Extract text from image using OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            return ""
    

