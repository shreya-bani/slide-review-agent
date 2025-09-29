"""
PDF file reader for extracting text content with page/paragraph locators.
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")
    
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


@dataclass
class PDFLocator:
    """Location information for PDF text elements."""
    page_index: int
    paragraph_index: int
    element_type: str  # 'heading', 'paragraph', 'bullet', 'caption'
    position: Optional[str] = None  # 'top', 'middle', 'bottom'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PDFExtractedText:
    """Text content with location information."""
    text: str
    locator: PDFLocator
    confidence: float = 1.0  # Confidence in text classification
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'locator': self.locator.to_dict(),
            'confidence': self.confidence
        }


@dataclass
class PDFPageContent:
    """Content from a single PDF page."""
    page_index: int
    raw_text: str
    elements: List[PDFExtractedText]
    page_size: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'page_index': self.page_index,
            'raw_text': self.raw_text,
            'elements': [elem.to_dict() for elem in self.elements],
            'page_size': self.page_size,
            'element_count': len(self.elements)
        }


class PDFReader:
    """Extract text content from PDF files."""
    
    def __init__(self):
        self.pdf_reader = None
        self.file_path = None
    
    def load_file(self, file_path: str) -> bool:
        """Load PDF file."""
        try:
            self.file_path = Path(file_path)
            self.pdf_reader = PdfReader(file_path)
            logger.info(f"Loaded PDF: {file_path} ({len(self.pdf_reader.pages)} pages)")
            return True
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            return False
    
    def extract_all_content(self) -> List[PDFPageContent]:
        """Extract content from all pages. Index start set as 1."""
        if not self.pdf_reader:
            raise ValueError("No PDF loaded")
        
        pages = []
        for idx, page in enumerate(self.pdf_reader.pages, start=1):
            page_content = self._extract_page_content(idx, page)
            pages.append(page_content)
        
        return pages
    
    def _extract_page_content(self, page_idx: int, page) -> PDFPageContent:
        """Extract content from a single page."""
        try:
            # Extract raw text
            raw_text = page.extract_text()
            
            # Get page dimensions
            page_size = None
            try:
                mediabox = page.mediabox
                page_size = (float(mediabox.width), float(mediabox.height))
            except:
                pass
            
            # Process text into elements
            elements = self._process_page_text(page_idx, raw_text)
            
            return PDFPageContent(
                page_index=page_idx,
                raw_text=raw_text,
                elements=elements,
                page_size=page_size
            )
        except Exception as e:
            logger.error(f"Error processing page {page_idx}: {e}")
            return PDFPageContent(
                page_index=page_idx,
                raw_text="",
                elements=[],
                page_size=None
            )
    
    def _process_page_text(self, page_idx: int, text: str) -> List[PDFExtractedText]:
        """Process raw text into structured elements."""
        if not text.strip():
            return []
        
        elements = []
        paragraphs = self._split_into_paragraphs(text)
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Classify text type
            element_type, confidence = self._classify_text(paragraph)
            
            # Estimate position
            position = self._estimate_position(para_idx, len(paragraphs))
            
            # Create locator
            locator = PDFLocator(
                page_index=page_idx,
                paragraph_index=para_idx,
                element_type=element_type,
                position=position
            )
            
            # Create element
            element = PDFExtractedText(
                text=paragraph.strip(),
                locator=locator,
                confidence=confidence
            )
            
            elements.append(element)
        
        return elements
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Split by double newlines
        paragraphs = text.split('\n\n')
        
        # Further process long paragraphs
        refined = []
        for para in paragraphs:
            refined.extend(self._refine_paragraph(para))
        
        return refined
    
    def _refine_paragraph(self, text: str) -> List[str]:
        """Refine paragraph splitting."""
        # Split very long paragraphs
        if len(text) > 1000:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            paragraphs = []
            current = ""
            
            for sentence in sentences:
                if len(current + sentence) > 500 and current:
                    paragraphs.append(current.strip())
                    current = sentence
                else:
                    current += " " + sentence if current else sentence
            
            if current:
                paragraphs.append(current.strip())
            
            return paragraphs
        
        # Split on bullet points
        if re.search(r'^\s*[•·▪▫‣⁃]\s+', text, re.MULTILINE):
            return re.split(r'\n(?=\s*[•·▪▫‣⁃]\s+)', text)
        
        # Split on numbered lists
        if re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
            return re.split(r'\n(?=\s*\d+\.\s+)', text)
        
        return [text]
    
    def _classify_text(self, text: str) -> Tuple[str, float]:
        """Classify text type and return confidence."""
        text = text.strip()
        
        # Check for headings
        if self._is_heading(text):
            return 'heading', 0.9
        
        # Check for bullet points
        if re.match(r'^\s*[•·▪▫‣⁃]\s+', text):
            return 'bullet', 0.95
        
        # Check for numbered lists
        if re.match(r'^\s*\d+\.\s+', text):
            return 'bullet', 0.9
        
        # Check for captions
        if (len(text) < 100 and 
            any(word in text.lower() for word in ['figure', 'table', 'chart', 'image'])):
            return 'caption', 0.8
        
        return 'paragraph', 0.7
    
    def _is_heading(self, text: str) -> bool:
        """Check if text looks like a heading."""
        text = text.strip()
        
        # Short text without ending punctuation
        if len(text) < 100 and not text.endswith(('.', '!', '?')):
            return True
        
        # All caps short text
        if len(text) < 50 and text.isupper():
            return True
        
        # Contains heading keywords
        heading_words = ['chapter', 'section', 'introduction', 'conclusion']
        if (len(text) < 80 and text[0].isupper() and 
            any(word in text.lower() for word in heading_words)):
            return True
        
        return False
    
    def _estimate_position(self, para_idx: int, total: int) -> str:
        """Estimate text position on page."""
        if total <= 1:
            return 'middle'
        
        if para_idx < total * 0.3:
            return 'top'
        elif para_idx > total * 0.7:
            return 'bottom'
        else:
            return 'middle'
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get document metadata."""
        if not self.pdf_reader:
            return {}
        
        metadata = {
            'file_path': str(self.file_path),
            'total_pages': len(self.pdf_reader.pages)
        }
        
        # Get document info
        try:
            if hasattr(self.pdf_reader, 'metadata') and self.pdf_reader.metadata:
                info = self.pdf_reader.metadata
                metadata.update({
                    'title': info.get('/Title'),
                    'author': info.get('/Author'),
                    'subject': info.get('/Subject'),
                    'creator': info.get('/Creator'),
                    'creation_date': str(info.get('/CreationDate')) if info.get('/CreationDate') else None
                })
        except:
            pass
        
        return metadata
    
    def extract_to_dict(self) -> Dict[str, Any]:
        """Extract all content to dictionary."""
        pages = self.extract_all_content()
        metadata = self.get_metadata()
        
        # Calculate stats
        total_elements = sum(len(page.elements) for page in pages)
        pages_with_content = len([p for p in pages if p.elements])
        
        type_counts = {}
        for page in pages:
            for element in page.elements:
                element_type = element.locator.element_type
                type_counts[element_type] = type_counts.get(element_type, 0) + 1
        
        return {
            'document_type': 'pdf',
            'metadata': metadata,
            'pages': [page.to_dict() for page in pages],
            'extraction_summary': {
                'total_pages': len(pages),
                'pages_with_content': pages_with_content,
                'total_elements': total_elements,
                'element_types': type_counts,
                'avg_elements_per_page': total_elements / len(pages) if pages else 0
            }
        }


def process_pdf_file(file_path: str) -> Dict[str, Any]:
    """Process a PDF file and return content."""
    reader = PDFReader()
    if not reader.load_file(file_path):
        raise ValueError(f"Failed to load PDF: {file_path}")
    return reader.extract_to_dict()


# if __name__ == "__main__":
#     import json
#     logging.basicConfig(level=logging.INFO)

#     # Example usage
#     try:
#         result = process_pdf_file(
#             r"C:\Users\Shreya\Documents\Projects\amida sled overview -feb 24 2025.pdf"
#         )

#         output_file = r'C:\Users\Shreya\Documents\GitHub\slide-review-agent\data\outputs\sample_pdf_result.json'
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)

#         print(f"JSON output saved to {output_file}")

#     except Exception as e:
#         print(f"Error: {e}")