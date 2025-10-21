"""
File Naming Convention Checker

This module checks document and folder names against Amida Technology Solutions'
naming conventions (ADMIN-POL-1-1) using an LLM-powered assistant.

The checker:
1. Loads the naming convention prompt from backend/prompts/file_naming_convention_prompt.md
2. Sends filename(s) to the LLM for analysis
3. Returns corrected naming options and explanations
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..services.llm_client import LLMClient
from ..utils.prompt_loader import load_prompt
from ..config.settings import settings

# LOGGER
logger = logging.getLogger(__name__)
logger.setLevel(settings.get_log_level())


class FileNamingChecker:
    """
    Checks file and folder names against Amida's naming conventions using LLM.
    """

    def __init__(self):
        """Initialize the file naming checker with LLM client."""
        self.llm = LLMClient()
        self.system_prompt = self._load_naming_prompt()

    def _load_naming_prompt(self) -> str:
        """Load the file naming convention prompt from the prompts directory."""
        try:
            prompt_content = load_prompt("file_naming_convention_prompt.md")
            logger.info("Loaded file naming convention prompt successfully")
            return prompt_content
        except FileNotFoundError as e:
            logger.error(f"Failed to load naming convention prompt: {e}")
            raise

    def check_filename(self, filename: str, concise: bool = False) -> Optional[str]:
        """
        Check a single filename against naming conventions.

        Args:
            filename: The filename to check
            concise: If True, request only the 4 corrected options without explanations

        Returns:
            LLM response with analysis and corrected options, or None if failed
        """
        if not filename or not filename.strip():
            logger.warning("Empty filename provided")
            return None

        if concise:
            user_message = (
                f"Provide ONLY the 4 corrected filename options for:\n\n{filename.strip()}\n\n"
                "Format your response as exactly 4 lines, one filename per line, with no additional text, "
                "explanations, or formatting. Just the filenames."
            )
        else:
            user_message = f"Please analyze this filename and provide corrected options:\n\n{filename.strip()}"

        try:
            response = self.llm.chat([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ])

            if response:
                logger.info(f"Successfully analyzed filename: {filename[:50]}...")
                return response
            else:
                logger.error(f"LLM returned no response for: {filename}")
                return None

        except Exception as e:
            logger.error(f"Error checking filename '{filename}': {e}")
            return None

    def check_multiple_filenames(self, filenames: List[str], concise: bool = False) -> Dict[str, Optional[str]]:
        """
        Check multiple filenames against naming conventions.

        Args:
            filenames: List of filenames to check
            concise: If True, request only the 4 corrected options without explanations

        Returns:
            Dictionary mapping each filename to its analysis result
        """
        results = {}

        for filename in filenames:
            if filename.strip():
                logger.info(f"Checking: {filename}")
                result = self.check_filename(filename, concise=concise)
                results[filename] = result
            else:
                logger.warning("Skipping empty filename")
                results[filename] = None

        return results

    def check_folder_name(self, folder_name: str) -> Optional[str]:
        """
        Check a folder name against naming conventions.

        Args:
            folder_name: The folder name to check

        Returns:
            LLM response with analysis and corrected options, or None if failed
        """
        if not folder_name or not folder_name.strip():
            logger.warning("Empty folder name provided")
            return None

        user_message = f"Please analyze this folder name and provide corrected options:\n\n{folder_name.strip()}"

        try:
            response = self.llm.chat([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ])

            if response:
                logger.info(f"Successfully analyzed folder name: {folder_name[:50]}...")
                return response
            else:
                logger.error(f"LLM returned no response for: {folder_name}")
                return None

        except Exception as e:
            logger.error(f"Error checking folder name '{folder_name}': {e}")
            return None

    def batch_check(self, items: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Check multiple items (files and/or folders) in batch.

        Args:
            items: List of dicts with keys 'name' and 'type' ('file' or 'folder')

        Returns:
            List of result dicts with keys 'name', 'type', 'analysis'
        """
        results = []

        for item in items:
            name = item.get("name", "")
            item_type = item.get("type", "file")

            if not name.strip():
                logger.warning(f"Skipping empty {item_type} name...")
                results.append({
                    "name": name,
                    "type": item_type,
                    "analysis": None,
                    "error": "Empty name provided"
                })
                continue

            logger.info(f"Checking {item_type}: {name}...")

            if item_type == "folder":
                analysis = self.check_folder_name(name)
            else:
                analysis = self.check_filename(name)

            results.append({
                "name": name,
                "type": item_type,
                "analysis": analysis,
                "error": None if analysis else "LLM returned no response!"
            })

        return results