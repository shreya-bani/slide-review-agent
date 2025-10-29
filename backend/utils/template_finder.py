"""
Template Finder Helper

Matches uploaded file names to their corresponding template JSON files.
Uses fuzzy matching and keyword detection to find the best template match.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class TemplateFinder:
    """
    Finds the best matching template for an uploaded document.

    Supports:
    - Exact name matching
    - Fuzzy name matching
    - Keyword-based matching
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize template finder.

        Args:
            template_dir: Directory containing template JSON files.
                         If None, uses default backend/templates/
        """
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            # Default to backend/templates/
            current_dir = Path(__file__).resolve().parent
            self.template_dir = current_dir.parent / "templates"

        self.templates = self._load_templates()
        logger.info(f"Loaded {len(self.templates)} templates from {self.template_dir}")

    def _load_templates(self) -> Dict[str, Path]:
        """
        Load all template JSON files from template directory.

        Returns:
            Dict mapping template name (without extension) to file path
        """
        templates = {}

        if not self.template_dir.exists():
            logger.warning(f"Template directory does not exist: {self.template_dir}")
            return templates

        for json_file in self.template_dir.glob("*.json"):
            template_name = json_file.stem.lower()  # Remove .json and lowercase
            templates[template_name] = json_file
            logger.debug(f"Found template: {template_name}")

        return templates

    def find_template(self, filename: str, threshold: float = 0.6) -> Optional[Path]:
        """
        Find the best matching template for the given filename.

        Args:
            filename: The uploaded file name (e.g., "my_ipr_document.pptx")
            threshold: Minimum similarity score (0.0 to 1.0) for fuzzy matching

        Returns:
            Path to the matching template JSON file, or None if no match found
        """
        if not self.templates:
            logger.warning("No templates available")
            return None

        # Clean the filename
        clean_name = self._clean_filename(filename)

        # Strategy 1: Exact match
        exact_match = self._exact_match(clean_name)
        if exact_match:
            logger.info(f"Exact match found: {exact_match.name}")
            return exact_match

        # Strategy 2: Keyword-based matching
        keyword_match = self._keyword_match(clean_name)
        if keyword_match:
            logger.info(f"Keyword match found: {keyword_match.name}")
            return keyword_match

        # Strategy 3: Fuzzy matching
        fuzzy_match, score = self._fuzzy_match(clean_name, threshold)
        if fuzzy_match:
            logger.info(f"Fuzzy match found: {fuzzy_match.name} (score: {score:.2f})")
            return fuzzy_match

        # Strategy 4: Default to standard template
        default_template = self._get_default_template()
        if default_template:
            logger.info(f"No specific match found for '{filename}', using default: {default_template.name}")
            return default_template

        logger.warning(f"No template match found for: {filename}")
        return None

    def _clean_filename(self, filename: str) -> str:
        """
        Clean filename for matching.

        Removes:
        - File extension
        - Special characters
        - Extra spaces
        - Numbers (optional)
        """
        # Remove extension
        name = Path(filename).stem

        # Convert to lowercase
        name = name.lower()

        # Replace underscores and hyphens with spaces
        name = name.replace('_', ' ').replace('-', ' ')

        # Remove common prefixes/suffixes
        remove_words = ['draft', 'final', 'v1', 'v2', 'v3', 'copy', 'new', 'updated']
        for word in remove_words:
            name = name.replace(word, '')

        # Remove extra spaces
        name = ' '.join(name.split())

        return name

    def _exact_match(self, clean_name: str) -> Optional[Path]:
        """
        Try exact template name match.

        Args:
            clean_name: Cleaned filename

        Returns:
            Template path if exact match found, None otherwise
        """
        # Check if clean name matches any template exactly
        if clean_name in self.templates:
            return self.templates[clean_name]

        return None

    def _keyword_match(self, clean_name: str) -> Optional[Path]:
        """
        Match based on key terms in the filename.

        Keywords define template types:
        - ipr, intellectual property → amida ipr template
        - itr, technical report → amida itr template
        - overview, project overview → amida project overview deck template
        - summary, project summary → amida project summary slide template
        - proposal, kick-off, kickoff → amida proposal kick-off meeting template
        - standard, general → amida standard template
        - table → amida table template

        Args:
            clean_name: Cleaned filename

        Returns:
            Template path if keyword match found, None otherwise
        """
        # Define keyword to template mapping
        keyword_map = {
            'amida ipr template': ['ipr', 'intellectual property', 'ip review'],
            'amida itr template': ['itr', 'technical report', 'tech report'],
            'amida project overview deck template': ['overview', 'project overview', 'deck'],
            'amida project summary slide template': ['summary', 'project summary'],
            'amida proposal kick-off meeting template': ['proposal', 'kick-off', 'kickoff', 'kick off'],
            'amida standard template': ['standard', 'general', 'default'],
            'amida table template': ['table', 'data table', 'tables'],
        }

        # Check each template's keywords
        for template_name, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in clean_name:
                    # Check if this template exists
                    if template_name in self.templates:
                        return self.templates[template_name]

        return None

    def _fuzzy_match(self, clean_name: str, threshold: float) -> Tuple[Optional[Path], float]:
        """
        Fuzzy string matching to find closest template.

        Args:
            clean_name: Cleaned filename
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            Tuple of (template_path, similarity_score) or (None, 0.0)
        """
        best_match = None
        best_score = 0.0

        for template_name, template_path in self.templates.items():
            # Calculate similarity ratio
            ratio = SequenceMatcher(None, clean_name, template_name).ratio()

            if ratio > best_score:
                best_score = ratio
                best_match = template_path

        if best_score >= threshold:
            return best_match, best_score

        return None, 0.0

    def _get_default_template(self) -> Optional[Path]:
        """
        Get the default template (amida standard template).

        Returns:
            Path to standard template, or None if not found
        """
        # Try to find standard template
        standard_names = [
            'amida standard template',
            'standard template',
            'amida standard',
            'standard'
        ]

        for name in standard_names:
            if name in self.templates:
                return self.templates[name]

        # If no standard template found, return the first available template
        if self.templates:
            logger.warning("Standard template not found, using first available template")
            return next(iter(self.templates.values()))

        return None

    def list_templates(self) -> List[str]:
        """
        Get list of available template names.

        Returns:
            List of template names (without .json extension)
        """
        return list(self.templates.keys())

    def get_template_path(self, template_name: str) -> Optional[Path]:
        """
        Get path for a specific template by name.

        Args:
            template_name: Name of template (with or without .json)

        Returns:
            Path to template file, or None if not found
        """
        # Remove .json if present
        clean_name = template_name.lower().replace('.json', '')

        return self.templates.get(clean_name)

    def load_template_json(self, template_path: Path) -> Optional[Dict]:
        """
        Load template JSON data.

        Args:
            template_path: Path to template JSON file

        Returns:
            Template data as dict, or None if loading fails
        """
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load template {template_path}: {e}")
            return None

    def find_and_load_template(self, filename: str, threshold: float = 0.6) -> Optional[Dict]:
        """
        Find and load template in one step.

        Args:
            filename: Uploaded file name
            threshold: Minimum similarity for fuzzy matching

        Returns:
            Template JSON data, or None if not found
        """
        template_path = self.find_template(filename, threshold)

        if template_path:
            return self.load_template_json(template_path)

        return None


def find_template_for_file(filename: str, template_dir: Optional[str] = None) -> Optional[Path]:
    """
    Convenience function to find template for a filename.

    Args:
        filename: Name of uploaded file
        template_dir: Optional custom template directory

    Returns:
        Path to matching template, or None
    """
    finder = TemplateFinder(template_dir)
    return finder.find_template(filename)


# if __name__ == "__main__":
#     import sys

#     # Test the template finder
#     print("=== Template Finder Test ===\n")

#     finder = TemplateFinder()

#     print(f"Template Directory: {finder.template_dir}")
#     print(f"Available Templates: {len(finder.list_templates())}")
#     for template in finder.list_templates():
#         print(f"  - {template}")

#     print("\n=== Testing File Matches ===\n")

#     # Test files
#     test_files = [
#         "my_ipr_document.pptx",
#         "Project_IPR_Review_2024.pptx",
#         "technical_report_v1.pptx",
#         "project_overview_deck_final.pptx",
#         "monthly_summary.pptx",
#         "kickoff_meeting_proposal.pptx",
#         "amida_standard_presentation.pptx",
#         "data_table_analysis.pptx",
#         "random_file_name.pptx",
#     ]

#     for test_file in test_files:
#         match = finder.find_template(test_file)
#         if match:
#             print(f"✓ {test_file}")
#             print(f"  → {match.name}")
#         else:
#             print(f"✗ {test_file}")
#             print(f"  → No match found")
#         print()

#     print("=== Custom File Test ===")
#     if len(sys.argv) > 1:
#         custom_file = sys.argv[1]
#         print(f"\nTesting: {custom_file}")
#         match = finder.find_template(custom_file)
#         if match:
#             print(f"Match: {match}")
#             template_data = finder.load_template_json(match)
#             if template_data:
#                 print(f"Template loaded successfully!")
#                 print(f"Slides in template: {len(template_data.get('slides', []))}")
#         else:
#             print("No match found")
