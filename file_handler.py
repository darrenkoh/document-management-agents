"""File handling operations for reading and moving files."""
import os
import shutil
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
    
    def __init__(self, source_path: str, destination_path: str):
        """Initialize file handler.
        
        Args:
            source_path: Source directory to read files from
            destination_path: Base destination directory for classified files
        """
        self.source_path = Path(source_path)
        self.destination_path = Path(destination_path)
        self.processed_files = set()
        
        # Ensure directories exist
        self.source_path.mkdir(parents=True, exist_ok=True)
        self.destination_path.mkdir(parents=True, exist_ok=True)
    
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
    
    def move_file(self, file_path: Path, category: str) -> bool:
        """Copy file to category-specific folder.
        
        Args:
            file_path: Path to the file to copy
            category: Classification category name
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Sanitize category name for filesystem
            safe_category = self._sanitize_category(category)
            
            # Create category directory
            category_dir = self.destination_path / safe_category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Destination file path
            dest_file = category_dir / file_path.name
            
            # Handle filename conflicts
            if dest_file.exists():
                base_name = file_path.stem
                suffix = file_path.suffix
                counter = 1
                while dest_file.exists():
                    new_name = f"{base_name}_{counter}{suffix}"
                    dest_file = category_dir / new_name
                    counter += 1
            
            # Copy file (preserves metadata with copy2)
            shutil.copy2(str(file_path), str(dest_file))
            logger.info(f"Copied {file_path.name} to {dest_file}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error copying file {file_path} to category {category}: {e}")
            return False
    
    def _sanitize_category(self, category: str) -> str:
        """Sanitize category name for use as directory name.
        
        Args:
            category: Category name
        
        Returns:
            Sanitized category name safe for filesystem
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        sanitized = category.strip()
        
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Replace multiple underscores with single
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "uncategorized"
        
        return sanitized
    
    def mark_processed(self, file_path: Path):
        """Mark a file as processed to avoid reprocessing."""
        self.processed_files.add(str(file_path))
    
    def is_processed(self, file_path: Path) -> bool:
        """Check if a file has been processed."""
        return str(file_path) in self.processed_files

