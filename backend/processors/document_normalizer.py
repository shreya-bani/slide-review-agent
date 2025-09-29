"""
Document normalizer to convert PPTX and PDF content to unified JSON schema.
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .pptx_reader import PPTXReader
from .pdf_reader import PDFReader

logger = logging.getLogger(__name__)


@dataclass
class UnifiedLocator:
    """Unified locator for both PPTX and PDF documents."""
    page_or_slide_index: int
    element_type: str  # 'title', 'heading', 'paragraph', 'bullet', 'notes', 'caption'
    element_index: int
    sub_index: Optional[int] = None
    position: Optional[str] = None  # 'top', 'middle', 'bottom'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedFormatting:
    """Unified formatting information."""
    font_name: Optional[str] = None
    font_size: Optional[int] = None
    is_bold: Optional[bool] = None
    is_italic: Optional[bool] = None
    color: Optional[str] = None
    hierarchy_level: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedTextElement:
    """Unified text element for both document types."""
    text: str
    locator: UnifiedLocator
    formatting: UnifiedFormatting
    confidence: float = 1.0
    source_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'locator': self.locator.to_dict(),
            'formatting': self.formatting.to_dict(),
            'confidence': self.confidence,
            'source_type': self.source_type
        }


@dataclass
class UnifiedPageContent:
    """Unified page/slide content."""
    index: int
    title: Optional[str] = None
    elements: List[UnifiedTextElement] = None
    notes: Optional[str] = None
    layout_info: Optional[str] = None
    
    def __post_init__(self):
        if self.elements is None:
            self.elements = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'title': self.title,
            'elements': [elem.to_dict() for elem in self.elements],
            'notes': self.notes,
            'layout_info': self.layout_info,
            'element_count': len(self.elements)
        }


@dataclass
class NormalizedDocument:
    """Unified document structure."""
    document_type: str
    metadata: Dict[str, Any]
    pages: List[UnifiedPageContent]
    extraction_info: Dict[str, Any]
    normalized_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_type': self.document_type,
            'metadata': self.metadata,
            'pages': [page.to_dict() for page in self.pages],
            'extraction_info': self.extraction_info,
            'normalized_at': self.normalized_at,
            'summary': {
                'total_pages': len(self.pages),
                'pages_with_content': len([p for p in self.pages if p.elements]),
                'total_elements': sum(len(p.elements) for p in self.pages)
            }
        }


class DocumentNormalizer:
    """Convert PPTX and PDF documents to unified format."""
    
    def __init__(self):
        self.pptx_reader = PPTXReader()
        self.pdf_reader = PDFReader()
    
    def normalize_document(self, file_path: str) -> NormalizedDocument:
        """Normalize a document (auto-detect type)."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.pptx':
            return self.normalize_pptx(str(file_path))
        elif extension == '.pdf':
            return self.normalize_pdf(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def normalize_pptx(self, file_path: str) -> NormalizedDocument:
        """Normalize a PowerPoint document."""
        if not self.pptx_reader.load_file(file_path):
            raise ValueError(f"Failed to load PPTX: {file_path}")
        
        raw_data = self.pptx_reader.extract_to_dict()
        
        # Convert to unified format
        unified_pages = []
        for slide_data in raw_data['slides']:
            unified_page = self._convert_pptx_slide(slide_data)
            unified_pages.append(unified_page)
        
        return NormalizedDocument(
            document_type='pptx',
            metadata=raw_data['metadata'],
            pages=unified_pages,
            extraction_info=raw_data['extraction_summary'],
            normalized_at=datetime.now().isoformat()
        )
    
    def normalize_pdf(self, file_path: str) -> NormalizedDocument:
        """Normalize a PDF document."""
        if not self.pdf_reader.load_file(file_path):
            raise ValueError(f"Failed to load PDF: {file_path}")
        
        raw_data = self.pdf_reader.extract_to_dict()
        
        # Convert to unified format
        unified_pages = []
        for page_data in raw_data['pages']:
            unified_page = self._convert_pdf_page(page_data)
            unified_pages.append(unified_page)
        
        return NormalizedDocument(
            document_type='pdf',
            metadata=raw_data['metadata'],
            pages=unified_pages,
            extraction_info=raw_data['extraction_summary'],
            normalized_at=datetime.now().isoformat()
        )
    
    def _convert_pptx_slide(self, slide_data: Dict[str, Any]) -> UnifiedPageContent:
        """Convert PPTX slide to unified format."""
        page = UnifiedPageContent(
            index=slide_data['slide_index'],
            title=slide_data.get('title'),
            notes=slide_data.get('notes'),
            layout_info=slide_data.get('layout_name')
        )
        
        # Convert content elements
        for elem_data in slide_data.get('content_elements', []):
            unified_elem = self._convert_pptx_element(elem_data)
            page.elements.append(unified_elem)
        
        return page
    
    def _convert_pptx_element(self, elem_data: Dict[str, Any]) -> UnifiedTextElement:
        """Convert PPTX text element to unified format."""
        locator_data = elem_data['locator']
        formatting_data = elem_data['formatting']
        
        # Create unified locator
        locator = UnifiedLocator(
            page_or_slide_index=locator_data['slide_index'],
            element_type=self._normalize_element_type(locator_data['element_type']),
            element_index=locator_data['element_index'],
            sub_index=locator_data.get('paragraph_index'),
            position=self._estimate_position_pptx(locator_data)
        )
        
        # Create unified formatting
        formatting = UnifiedFormatting(
            font_name=formatting_data.get('font_name'),
            font_size=formatting_data.get('font_size'),
            is_bold=formatting_data.get('is_bold'),
            is_italic=formatting_data.get('is_italic'),
            color=formatting_data.get('font_color'),
            hierarchy_level=locator_data.get('bullet_level')
        )
        
        return UnifiedTextElement(
            text=elem_data['text'],
            locator=locator,
            formatting=formatting,
            confidence=1.0,
            source_type='pptx'
        )
    
    def _convert_pdf_page(self, page_data: Dict[str, Any]) -> UnifiedPageContent:
        """Convert PDF page to unified format."""
        page = UnifiedPageContent(
            index=page_data['page_index'],
            title=self._extract_pdf_title(page_data),
            notes=None,  # PDFs don't have speaker notes
            layout_info=f"Page size: {page_data.get('page_size', 'Unknown')}"
        )
        
        # Convert content elements
        for elem_data in page_data.get('elements', []):
            unified_elem = self._convert_pdf_element(elem_data)
            page.elements.append(unified_elem)
        
        return page
    
    def _convert_pdf_element(self, elem_data: Dict[str, Any]) -> UnifiedTextElement:
        """Convert PDF text element to unified format."""
        locator_data = elem_data['locator']
        
        # Create unified locator
        locator = UnifiedLocator(
            page_or_slide_index=locator_data['page_index'],
            element_type=self._normalize_element_type(locator_data['element_type']),
            element_index=locator_data['paragraph_index'],
            sub_index=None,
            position=locator_data.get('position')
        )
        
        # Create unified formatting (PDFs have limited formatting info)
        formatting = UnifiedFormatting(
            font_name=None,
            font_size=None,
            is_bold=None,
            is_italic=None,
            color=None,
            hierarchy_level=None
        )
        
        return UnifiedTextElement(
            text=elem_data['text'],
            locator=locator,
            formatting=formatting,
            confidence=elem_data.get('confidence', 1.0),
            source_type='pdf'
        )
    
    def _normalize_element_type(self, original_type: str) -> str:
        """Normalize element types to common vocabulary."""
        type_mapping = {
            # PPTX types
            'title': 'title',
            'content': 'paragraph',
            'bullet': 'bullet',
            'notes': 'notes',
            
            # PDF types
            'heading': 'heading',
            'paragraph': 'paragraph',
            'bullet': 'bullet',
            'caption': 'caption'
        }
        
        return type_mapping.get(original_type, 'paragraph')
    
    def _estimate_position_pptx(self, locator_data: Dict[str, Any]) -> str:
        """Estimate position for PPTX elements."""
        para_idx = locator_data.get('paragraph_index', 0)
        
        if para_idx == 0:
            return 'top'
        elif para_idx <= 2:
            return 'middle'
        else:
            return 'bottom'
    
    def _extract_pdf_title(self, page_data: Dict[str, Any]) -> Optional[str]:
        """Extract title from PDF page elements."""
        elements = page_data.get('elements', [])
        
        for elem in elements:
            if elem['locator']['element_type'] == 'heading':
                return elem['text']
        
        # If no heading found, check for first element if it looks like a title
        if elements:
            first_elem = elements[0]
            text = first_elem['text']
            if len(text) < 100 and not text.endswith('.'):
                return text
        
        return None


def normalize_document_file(file_path: str) -> Dict[str, Any]:
    """Convenience function to normalize a document file."""
    normalizer = DocumentNormalizer()
    normalized_doc = normalizer.normalize_document(file_path)
    return normalized_doc.to_dict()


def normalize_pptx_file(file_path: str) -> Dict[str, Any]:
    """Convenience function to normalize a PPTX file."""
    normalizer = DocumentNormalizer()
    normalized_doc = normalizer.normalize_pptx(file_path)
    return normalized_doc.to_dict()


def normalize_pdf_file(file_path: str) -> Dict[str, Any]:
    """Convenience function to normalize a PDF file."""
    normalizer = DocumentNormalizer()
    normalized_doc = normalizer.normalize_pdf(file_path)
    return normalized_doc.to_dict()


# if __name__ == "__main__":
#     import json

#     # Example usage
#     try:
#         # Test with a sample file
#         input_file = r'C:\Users\Shreya\Documents\Projects\Amida Agentic AI solution Strategic Plan Draft Aug 28 2025 maf.pptx'
#         result = normalize_document_file(input_file)

#         output_file = r'C:\Users\Shreya\Documents\GitHub\slide-review-agent\data\outputs\sample_normalized_pptx_result.json'
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)

#         print(f"JSON output saved to {output_file}")

#     except Exception as e:
#         print(f"Error: {e}")