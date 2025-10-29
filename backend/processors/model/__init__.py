"""
Models package for document processors.
"""
from .pptx_models import (
    TextLocator,
    EnhancedFormattingInfo,
    TableCell,
    ExtractedTable,
    ExtractedText,
    SlideContent,
    ThemeInfo
)

__all__ = [
    'TextLocator',
    'EnhancedFormattingInfo',
    'TableCell',
    'ExtractedTable',
    'ExtractedText',
    'SlideContent',
    'ThemeInfo'
]
