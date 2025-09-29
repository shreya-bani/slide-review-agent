"""
PowerPoint file reader for extracting text content with precise locators.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from pptx import Presentation
from pptx.shapes.base import BaseShape
from pptx.text.text import TextFrame

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


@dataclass
class TextLocator:
    """Precise location information for text elements."""
    slide_index: int
    element_type: str  # 'title', 'content', 'bullet', 'notes'
    element_index: int
    paragraph_index: Optional[int] = None
    bullet_level: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FormattingInfo:
    """Formatting information for text elements."""
    font_name: Optional[str] = None
    font_size: Optional[int] = None
    is_bold: Optional[bool] = None
    is_italic: Optional[bool] = None
    font_color: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedText:
    """Text content with location and formatting information."""
    text: str
    locator: TextLocator
    formatting: FormattingInfo
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'locator': self.locator.to_dict(),
            'formatting': self.formatting.to_dict()
        }


@dataclass
class SlideContent:
    """Complete content from a single slide."""
    slide_index: int
    title: Optional[str] = None
    content_elements: List[ExtractedText] = None
    notes: Optional[str] = None
    layout_name: Optional[str] = None
    
    def __post_init__(self):
        if self.content_elements is None:
            self.content_elements = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'slide_index': self.slide_index,
            'title': self.title,
            'content_elements': [elem.to_dict() for elem in self.content_elements],
            'notes': self.notes,
            'layout_name': self.layout_name,
            'element_count': len(self.content_elements)
        }


class PPTXReader:
    """Extract text content from PowerPoint files."""
    
    def __init__(self):
        self.presentation = None
        self.file_path = None
    
    def load_file(self, file_path: str) -> bool:
        """Load PowerPoint file."""
        try:
            self.file_path = Path(file_path)
            self.presentation = Presentation(file_path)
            logger.info(f"Loaded PPTX: {file_path} ({len(self.presentation.slides)} slides)")
            return True
        except Exception as e:
            logger.error(f"Failed to load PPTX {file_path}: {e}")
            return False
    
    def extract_all_content(self) -> List[SlideContent]:
        """Extract content from all slides (1-based indexing)."""
        if not self.presentation:
            raise ValueError("No presentation loaded")
        
        slides = []
        # enumerate starting at 1
        for idx, slide in enumerate(self.presentation.slides, start=1):
            slide_content = self._extract_slide_content(idx, slide)
            slides.append(slide_content)
        
        return slides
    
    def _extract_slide_content(self, slide_idx: int, slide) -> SlideContent:
        """Extract content from a single slide."""
        slide_content = SlideContent(slide_index=slide_idx)
        
        # Get layout name
        try:
            slide_content.layout_name = slide.slide_layout.name
        except:
            slide_content.layout_name = "Unknown"
        
        # Extract title
        slide_content.title = self._extract_title(slide)
        
        # Extract content from shapes
        slide_content.content_elements = self._extract_shapes_content(slide_idx, slide)
        
        # Extract notes
        slide_content.notes = self._extract_notes(slide)
        
        return slide_content
    
    def _extract_title(self, slide) -> Optional[str]:
        """Extract slide title."""
        try:
            if hasattr(slide, 'shapes') and slide.shapes.title:
                title = slide.shapes.title.text.strip()
                return title if title else None
        except:
            pass
        return None
    
    def _extract_shapes_content(self, slide_idx: int, slide) -> List[ExtractedText]:
        """Extract text from all shapes in slide."""
        elements = []
        content_idx = 0
        
        for shape in slide.shapes:
            if not hasattr(shape, 'text_frame') or not shape.text_frame:
                continue
            
            # Skip title (already extracted)
            if shape == slide.shapes.title:
                continue
            
            shape_elements = self._extract_text_frame(slide_idx, shape.text_frame, content_idx)
            elements.extend(shape_elements)
            content_idx += 1
        
        return elements
    
    def _extract_text_frame(self, slide_idx: int, text_frame: TextFrame, 
                           shape_idx: int) -> List[ExtractedText]:
        """Extract text from a text frame."""
        elements = []
        
        for para_idx, paragraph in enumerate(text_frame.paragraphs):
            text = paragraph.text.strip()
            if not text:
                continue
            
            # Get bullet level
            bullet_level = getattr(paragraph, 'level', None)
            
            # Get formatting info
            formatting = self._extract_formatting(paragraph)
            
            # Determine element type
            element_type = 'bullet' if bullet_level is not None else 'content'
            
            # Create locator
            locator = TextLocator(
                slide_index=slide_idx,
                element_type=element_type,
                element_index=shape_idx,
                paragraph_index=para_idx,
                bullet_level=bullet_level
            )
            
            # Create extracted text
            element = ExtractedText(
                text=text,
                locator=locator,
                formatting=formatting
            )
            
            elements.append(element)
        
        return elements
    
    def _extract_formatting(self, paragraph) -> FormattingInfo:
        """Extract formatting information from paragraph."""
        formatting = FormattingInfo()
        
        try:
            if paragraph.runs:
                run = paragraph.runs[0]
                font = run.font
                
                formatting.font_name = font.name
                formatting.font_size = int(font.size.pt) if font.size else None
                formatting.is_bold = font.bold
                formatting.is_italic = font.italic
                
                # Extract color if available
                if hasattr(font, 'color') and font.color:
                    try:
                        rgb = font.color.rgb
                        formatting.font_color = f"rgb({rgb.r},{rgb.g},{rgb.b})"
                    except:
                        pass
        except Exception as e:
            logger.debug(f"Could not extract formatting: {e}")
        
        return formatting
    
    def _extract_notes(self, slide) -> Optional[str]:
        """Extract speaker notes."""
        try:
            if hasattr(slide, 'notes_slide') and slide.notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                return notes if notes else None
        except:
            pass
        return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get document metadata."""
        if not self.presentation:
            return {}
        
        metadata = {
            'file_path': str(self.file_path),
            'total_slides': len(self.presentation.slides),
            'layouts': []
        }
        
        # Get unique layouts
        layouts = set()
        for slide in self.presentation.slides:
            try:
                layouts.add(slide.slide_layout.name)
            except:
                layouts.add("Unknown")
        metadata['layouts'] = list(layouts)
        
        # Get document properties
        try:
            props = self.presentation.core_properties
            metadata.update({
                'title': props.title,
                'author': props.author,
                'created': props.created.isoformat() if props.created else None,
                'modified': props.modified.isoformat() if props.modified else None
            })
        except:
            pass
        
        return metadata
    
    def extract_to_dict(self) -> Dict[str, Any]:
        """Extract all content to dictionary."""
        slides = self.extract_all_content()
        metadata = self.get_metadata()
        
        return {
            'document_type': 'pptx',
            'metadata': metadata,
            'slides': [slide.to_dict() for slide in slides],
            'extraction_summary': {
                'total_slides': len(slides),
                'slides_with_content': len([s for s in slides if s.content_elements]),
                'slides_with_notes': len([s for s in slides if s.notes]),
                'total_elements': sum(len(s.content_elements) for s in slides)
            }
        }


def process_pptx_file(file_path: str) -> Dict[str, Any]:
    """Process a PPTX file and return content."""
    reader = PPTXReader()
    if not reader.load_file(file_path):
        raise ValueError(f"Failed to load PPTX: {file_path}")
    return reader.extract_to_dict()


# if __name__ == "__main__":
#     import json

#     # Example usage
#     try:
#         result = process_pptx_file(
#             r"C:\Users\Shreya\Documents\Projects\Amida Agentic AI solution Strategic Plan Draft Aug 28 2025 maf.pptx"
#         )

#         output_file = r'C:\Users\Shreya\Documents\GitHub\slide-review-agent\data\outputs\sample_pptx_result.json'
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)

#         print(f"JSON output saved to {output_file}")

#     except Exception as e:
#         print(f"Error: {e}")
