from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config.settings import settings
from ..services.llm_client import LLMClient
from .protection_layer import ProtectionLayer, LLMConfigError
from .grammar_checker import GrammarChecker
from .word_list_checker import WordListChecker
from .tone_analyzer import AdvancedStyleAnalyzer
from .usage_analyzer import UsageAnalyzer
from .model.models import StyleIssue

logger = logging.getLogger(__name__)

class AgenticStyleChecker:
    """
    Orchestrates protection-layer + GrammarChecker + WordListChecker + ToneAnalyzer.
    All analyzers share the same protection data for consistency and efficiency.
    """
    def __init__(self, use_llm: bool = True, include_tone: bool = True, include_usage: bool = True):
        self.llm = LLMClient() if use_llm else None
        self.protection_layer = ProtectionLayer(llm_client=self.llm)
        self.protection_data: Dict[str, Any] = {}
        self.include_tone = include_tone
        self.include_usage = include_usage
        self.stats = {
            "total_elements": 0,
            "protected_items": {},
            "by_severity": {"error": 0, "warning": 0, "suggestion": 0, "info": 0},
            "by_category": {"grammar": 0, "word-list": 0, "tone": 0, "usage": 0},
        }

    def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("STYLE CHECKER - Starting document analysis")

        # Protection cache by doc_id
        meta = document.get("metadata", {}) or {}
        doc_id = meta.get("doc_id") or meta.get("file_id") or meta.get("id")
        cache_path = Path(settings.output_dir) / f"{doc_id}_protection.json" if doc_id else None

        loaded = False
        if cache_path:
            try:
                loaded = self.protection_layer.load(cache_path)
                if loaded:
                    self.protection_data = self.protection_layer.data
                    logger.info("Protection cache loaded for doc_id=%s from %s", doc_id, cache_path)
            except Exception as e:
                logger.warning("Protection cache load failed (%s). Will re-detect.", e)

        if not loaded:
            try:
                self.protection_data = self.protection_layer.detect_all_protected_content(document)
                if cache_path:
                    try:
                        self.protection_layer.save(cache_path)
                        logger.info("Protection cache saved for doc_id=%s to %s", doc_id, cache_path)
                    except Exception as e:
                        logger.warning("Failed to save protection cache to %s: %s", cache_path, e)
            except LLMConfigError as e:
                logger.error("Protection detection skipped: %s", e)
                self.protection_data = {}

        # Log protection stats
        total_protected = 0
        for category, items in self.protection_data.items():
            count = len(items)
            self.stats["protected_items"][category] = count
            total_protected += count
            logger.info("> %s: %d items", category, count)
        logger.info("Total protected items: %d", total_protected)

        # Apply grammar & word list
        logger.info("Running grammar and word-list checks...")
        grammar_checker = GrammarChecker(self.protection_data)
        word_list_checker = WordListChecker(self.protection_data)

        all_issues: List[StyleIssue] = []
        for page in document.get("pages", []):
            slide_idx = page.get("index", 0)
            for elem in page.get("elements", []):
                text = (elem.get("text") or "").strip()
                if not text or len(text) < 3:
                    continue
                elem_idx = (elem.get("locator") or {}).get("element_index", 0)
                self.stats["total_elements"] += 1
                all_issues.extend(grammar_checker.check(text, elem, slide_idx, elem_idx))
                all_issues.extend(word_list_checker.check(text, elem, slide_idx, elem_idx))

        logger.info(f"Grammar and word-list checks complete: {len(all_issues)} issues found")

        # Apply tone analysis if enabled
        tone_issues_dicts = []
        if self.include_tone:
            logger.info("Running tone analysis...")
            tone_analyzer = AdvancedStyleAnalyzer(use_llm=True, protection_data=self.protection_data)
            tone_result = tone_analyzer.analyze(document)
            tone_issues_dicts = tone_result.get("issues", [])
            logger.info(f"Tone analysis complete: {len(tone_issues_dicts)} issues found")

        # Apply usage analysis if enabled
        usage_issues_dicts = []
        if self.include_usage:
            logger.info("Running usage analysis...")
            usage_analyzer = UsageAnalyzer(use_llm=True, protection_data=self.protection_data)
            usage_result = usage_analyzer.analyze(document)
            usage_issues_dicts = usage_result.get("issues", [])
            logger.info(f"Usage analysis complete: {len(usage_issues_dicts)} issues found")

        # Combine all issues
        issues_dicts = [i.to_dict() for i in all_issues] + tone_issues_dicts + usage_issues_dicts

        # Fill tallies
        for issue in issues_dicts:
            self.stats["by_severity"][issue["severity"]] += 1
            self.stats["by_category"][issue["category"]] += 1

        return {
            "issues": issues_dicts,
            "statistics": self.stats,
            "protection_data": self.protection_data,
            "total_issues": len(issues_dicts),
            "document_metadata": document.get("metadata", {}),
        }


def check_document(
    document: Dict[str, Any],
    *,
    use_llm: bool = True,
    include_tone: bool = True,
    include_usage: bool = True,
    return_wrapper: bool = False,
    precomputed_protection: Optional[Dict[str, Any]] = None,
):
    """
    Check document for style issues using grammar, word-list, and optionally tone and usage analyzers.

    Args:
        document: Normalized document to analyze
        use_llm: Whether to use LLM for protection detection (default: True)
        include_tone: Whether to include tone analysis (default: True)
        include_usage: Whether to include usage analysis (default: True)
        return_wrapper: Return full result dict or just issues list (default: False)
        precomputed_protection: Pre-computed protection data to reuse (default: None)

    Returns:
        Full result dictionary or issues list depending on return_wrapper
    """
    checker = AgenticStyleChecker(use_llm=use_llm, include_tone=include_tone, include_usage=include_usage)

    if precomputed_protection is not None:
        # Inject precomputed protection; bypass LLM calls
        checker.protection_data = precomputed_protection
        def _noop_detect(_doc: Dict[str, Any]) -> Dict[str, Any]:
            return precomputed_protection
        checker.protection_layer.detect_all_protected_content = _noop_detect  # type: ignore

    result = checker.analyze_document(document)
    return result if return_wrapper else result.get("issues", [])
