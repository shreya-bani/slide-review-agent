"""
Data models for PPTX extraction.

This module contains all dataclass definitions used for extracting and
representing content from PowerPoint presentations.
"""
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any


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
class EnhancedFormattingInfo:
    """Comprehensive formatting information for text elements."""
    # Font properties
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    is_bold: Optional[bool] = None
    is_italic: Optional[bool] = None
    is_underline: Optional[bool] = None

    # Font color (multiple representations)
    font_color: Optional[str] = None  # Display format
    font_color_rgb: Optional[str] = None  # Hex RGB
    font_color_theme: Optional[str] = None  # Theme color name
    font_color_brightness: Optional[float] = None  # Theme brightness modifier

    # Paragraph spacing (points)
    paragraph_spacing_before: Optional[float] = None
    paragraph_spacing_after: Optional[float] = None
    line_spacing: Optional[float] = None

    # Paragraph layout
    paragraph_alignment: Optional[str] = None
    left_margin: Optional[float] = None
    right_margin: Optional[float] = None
    first_line_indent: Optional[float] = None

    # Bullet formatting
    bullet_type: Optional[str] = None  # 'numbered', 'bulleted', 'none'
    bullet_char: Optional[str] = None
    bullet_font: Optional[str] = None
    bullet_size: Optional[float] = None

    # Background
    background_color: Optional[str] = None
    background_color_rgb: Optional[str] = None

    # Master slide info
    inherited_from_master: bool = False
    master_slide_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TableCell:
    """Individual table cell with content and formatting."""
    row_index: int
    col_index: int
    text: str
    formatting: EnhancedFormattingInfo
    is_header: bool = False
    rowspan: int = 1
    colspan: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'row_index': self.row_index,
            'col_index': self.col_index,
            'text': self.text,
            'formatting': self.formatting.to_dict(),
            'is_header': self.is_header,
            'rowspan': self.rowspan,
            'colspan': self.colspan
        }


@dataclass
class ExtractedTable:
    """Table structure with cells and metadata."""
    slide_index: int
    table_index: int
    rows: int
    columns: int
    cells: List[TableCell] = field(default_factory=list)
    has_header_row: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'slide_index': self.slide_index,
            'table_index': self.table_index,
            'rows': self.rows,
            'columns': self.columns,
            'cells': [cell.to_dict() for cell in self.cells],
            'has_header_row': self.has_header_row
        }


@dataclass
class ExtractedText:
    """Text content with location and formatting information."""
    text: str
    locator: TextLocator
    formatting: EnhancedFormattingInfo

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
    content_elements: List[ExtractedText] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    notes: Optional[str] = None
    layout_name: Optional[str] = None
    master_name: Optional[str] = None
    background_color: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'slide_index': self.slide_index,
            'title': self.title,
            'content_elements': [elem.to_dict() for elem in self.content_elements],
            'tables': [table.to_dict() for table in self.tables],
            'notes': self.notes,
            'layout_name': self.layout_name,
            'master_name': self.master_name,
            'background_color': self.background_color,
            'element_count': len(self.content_elements),
            'table_count': len(self.tables)
        }


@dataclass
class ThemeInfo:
    """Theme information extracted from presentation."""
    theme_name: Optional[str] = None
    color_scheme: Dict[str, str] = field(default_factory=dict)
    font_scheme: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
