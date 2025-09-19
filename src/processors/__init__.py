"""
Document processors package
"""

from .pptx_processor import extract_text_from_pptx
from .document_processor import process_document

__all__ = ['extract_text_from_pptx', 'process_document']