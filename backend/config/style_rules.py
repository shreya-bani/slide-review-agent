"""
Amida Style Guide Rules Configuration
Based on Amida Style Guide January 2025 and Documents/Folder Naming Convention Policy
"""
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class StyleRule:
    """Individual style rule definition."""
    name: str
    category: str  # 'voice', 'formatting', 'grammar', 'inclusivity', 'naming'
    severity: str  # 'critical', 'warning', 'suggestion'
    description: str
    guide_section: str  # Reference to Amida Style Guide section
    pattern: str = None  # Regex pattern if applicable
    examples: Dict[str, str] = None  # {'incorrect': '...', 'correct': '...'}

class AmidaStyleRules:
    """Complete Amida Style Guide rules implementation."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.word_list = self._initialize_word_list()
        self.heading_styles = self._initialize_heading_styles()
    
    def _initialize_rules(self) -> List[StyleRule]:
        """Initialize all style rules from the Amida Style Guide."""
        return [
            # VOICE RULES
            StyleRule(
                name="active_voice",
                category="voice",
                severity="warning",
                description="Use active voice instead of passive voice",
                guide_section="II.A - Active Voice",
                examples={
                    "incorrect": "Amida was founded in 2013",
                    "correct": "Matt and Peter co-founded Amida in 2013"
                }
            ),
            StyleRule(
                name="positive_language",
                category="voice",
                severity="suggestion",
                description="Frame statements positively rather than negatively",
                guide_section="II.B - Positive Language",
                examples={
                    "incorrect": "This procedure is not difficult to follow",
                    "correct": "This procedure is easy to follow"
                }
            ),
            StyleRule(
                name="avoid_mushy_language",
                category="voice",
                severity="warning",
                description="Use direct, clear language instead of vague constructions",
                guide_section="II.A - Active Voice",
                pattern=r"\bare\s+(looking\s+for|seeking|trying\s+to)",
                examples={
                    "incorrect": "We are looking for a few good engineers who have expertise in...",
                    "correct": "We hire engineers with expertise in..."
                }
            ),
            
            # SPECIFICITY RULES
            StyleRule(
                name="avoid_temporal_vagueness",
                category="voice",
                severity="critical",
                description="Avoid vague temporal phrases like 'currently' or 'at this time'",
                guide_section="III.A - Specificity",
                pattern=r"\b(currently|at this time|presently|right now)\b",
                examples={
                    "incorrect": "The system is currently undergoing maintenance",
                    "correct": "The system is undergoing maintenance as of January 2025"
                }
            ),
            StyleRule(
                name="specific_people_terms",
                category="voice",
                severity="suggestion",
                description="Use specific terms instead of generic 'people'",
                guide_section="III.A - Specificity",
                pattern=r"\bpeople\b(?!\s+(instead|rather))",
                examples={
                    "incorrect": "People will use this system",
                    "correct": "End users will access this system"
                }
            ),
            
            # INCLUSIVITY RULES
            StyleRule(
                name="singular_they",
                category="inclusivity",
                severity="suggestion",
                description="Use 'they' as acceptable gender-neutral singular pronoun",
                guide_section="III.B - Inclusivity",
                examples={
                    "incorrect": "Each user should check his account",
                    "correct": "Each user should check their account"
                }
            ),
            StyleRule(
                name="avoid_gendered_terms",
                category="inclusivity",
                severity="warning",
                description="Use gender-neutral terms",
                guide_section="III.B - Inclusivity",
                pattern=r"\b(waitress|guys|chairman|mankind)\b",
                examples={
                    "incorrect": "server (waitress), guys (people)",
                    "correct": "server, people"
                }
            ),
            StyleRule(
                name="person_first_language",
                category="inclusivity",
                severity="critical",
                description="Center the person, not the condition when referencing disabilities",
                guide_section="III.B - Inclusivity",
                examples={
                    "incorrect": "He is disabled",
                    "correct": "She has a disability"
                }
            ),
            
            # GRAMMAR RULES
            StyleRule(
                name="acronym_expansion",
                category="grammar",
                severity="critical",
                description="Write out full phrase before using acronym",
                guide_section="IV.A - Acronyms",
                pattern=r"\b[A-Z]{2,}\b",
                examples={
                    "incorrect": "The DRE will benefit VA",
                    "correct": "The Data Reconciliation Engine (DRE) will benefit the U.S. Department of Veterans Affairs (VA)"
                }
            ),
            StyleRule(
                name="bullet_formatting",
                category="formatting",
                severity="warning",
                description="No periods at ends of bullet points, use parallel construction",
                guide_section="IV.B - Bullets",
                examples={
                    "incorrect": "• First item.\n• Second item.",
                    "correct": "• First item\n• Second item"
                }
            ),
            StyleRule(
                name="oxford_comma",
                category="grammar",
                severity="critical",
                description="Always include Oxford comma in lists of three or more",
                guide_section="VI - Mechanics",
                pattern=r"(\w+),\s(\w+)\sand\s(\w+)",
                examples={
                    "incorrect": "Peter asked for updates on Indaba, Orange Rx and DRE",
                    "correct": "Peter asked for updates on Indaba, Orange Rx, and DRE"
                }
            ),
            StyleRule(
                name="en_dash_usage",
                category="grammar",
                severity="suggestion",
                description="Use en dashes (–) not em dashes (—) or double hyphens (--)",
                guide_section="IV.D - Hyphens",
                pattern=r"(—|--)",
                examples={
                    "incorrect": "Our first step was to establish the purpose -- or 'why' -- of the plan",
                    "correct": "Our first step was to establish the purpose – or 'why' – of the plan"
                }
            ),
            StyleRule(
                name="numeral_spelling",
                category="grammar",
                severity="suggestion",
                description="Spell out numbers below 100 in non-technical writing",
                guide_section="IV.E - Numerals",
                pattern=r"\b([1-9][0-9]?)\b(?!\s*(percent|%|million|billion))",
                examples={
                    "incorrect": "We responded to 50 opportunities",
                    "correct": "We responded to fifty opportunities"
                }
            ),
            StyleRule(
                name="single_space_after_period",
                category="formatting",
                severity="warning",
                description="Use only one space after periods",
                guide_section="IV.F - Periods",
                pattern=r"\.\s{2,}",
                examples={
                    "incorrect": "First sentence.  Second sentence.",
                    "correct": "First sentence. Second sentence."
                }
            ),
            StyleRule(
                name="title_case",
                category="formatting",
                severity="critical",
                description="Use Title Case for document titles",
                guide_section="IV.H - Titles",
                examples={
                    "incorrect": "amida causal analysis framework",
                    "correct": "Amida Causal Analysis Framework"
                }
            ),
            
            # FORMAL WRITING RULES
            StyleRule(
                name="avoid_contractions",
                category="grammar",
                severity="critical",
                description="Do not use contractions in formal writing",
                guide_section="VI - Mechanics",
                pattern=r"\b\w+[''](?:t|re|ll|ve|d)\b",
                examples={
                    "incorrect": "We don't use contractions",
                    "correct": "We do not use contractions"
                }
            ),
            StyleRule(
                name="avoid_and_or",
                category="grammar",
                severity="warning",
                description="Do not use 'and/or' in formal writing",
                guide_section="VI - Mechanics",
                pattern=r"\band/or\b",
                examples={
                    "incorrect": "Submit documents and/or reports",
                    "correct": "Submit documents or reports"
                }
            ),
            StyleRule(
                name="avoid_ampersand",
                category="formatting",
                severity="warning",
                description="Avoid '&' in formal writing",
                guide_section="VI - Mechanics",
                pattern=r"\s&\s",
                examples={
                    "incorrect": "Research & Development",
                    "correct": "Research and Development"
                }
            ),
            StyleRule(
                name="avoid_etc",
                category="grammar",
                severity="suggestion",
                description="Avoid 'etc.' in formal writing",
                guide_section="VI - Mechanics",
                pattern=r"\betc\.?\b",
                examples={
                    "incorrect": "Tools, processes, etc.",
                    "correct": "Tools, processes, and other resources"
                }
            ),
            StyleRule(
                name="avoid_second_person",
                category="grammar",
                severity="critical",
                description="Avoid second-person pronouns ('you', 'yours') in formal writing",
                guide_section="VI - Mechanics",
                pattern=r"\b(you|your|yours)\b",
                examples={
                    "incorrect": "You should submit your report",
                    "correct": "Staff should submit their reports"
                }
            ),
            StyleRule(
                name="avoid_unnecessary_gerunds",
                category="grammar",
                severity="suggestion",
                description="Avoid unnecessary gerunds (-ing forms)",
                guide_section="VI - Mechanics",
                pattern=r"\b(avoid using|start using|begin using)\b",
                examples={
                    "incorrect": "Avoid using contractions",
                    "correct": "Do not use contractions"
                }
            ),
            
            # HANGING WORDS RULE
            StyleRule(
                name="no_hanging_words",
                category="formatting",
                severity="warning",
                description="No single words hanging alone on their own line in slide decks",
                guide_section="VI - Mechanics",
                examples={
                    "incorrect": "This is a very long bullet point that ends with a single\nword",
                    "correct": "This is a very long bullet point that ends\nwith proper line breaks"
                }
            ),
        ]
    
    def _initialize_word_list(self) -> Dict[str, str]:
        """Amida's specific word usage preferences from Section V."""
        return {
            # Compound words
            "built in": "built-in",
            "cyber-attacks": "cyberattacks", 
            "cyber security": "cybersecurity",
            "cyber-security": "cybersecurity",
            "code-sets": "code sets",
            "codesets": "code sets",
            "health care": "healthcare",
            "public-sector": "public sector",
            "user friendly": "user-friendly",
            "web based": "web-based",
            
            # Noun/adjective forms
            "end-user": {"noun": "end user", "adjective": "end-user"},
            "open-source": {"noun": "open source", "adjective": "open-source"},
            "set-up": {"verb": "set up", "noun": "set-up"},
            
            # Capitalization
            "machine learning": "Machine Learning",
            "artificial intelligence": "Artificial Intelligence", 
            "natural language processing": "Natural Language Processing",
            
            # Latin abbreviations
            "e.g.": "e.g., (always followed by comma)",
            "i.e.": "i.e., (always followed by comma)",
        }
    
    def _initialize_heading_styles(self) -> Dict[str, Dict[str, Any]]:
        """Heading style requirements from Section IV.C."""
        return {
            "Heading 1": {
                "font": "Lato Bold",
                "color": "RGB 38-111-160",
                "size": 20,
                "spacing": {"before": 0, "after": 12}
            },
            "Heading 2": {
                "font": "Lato Regular", 
                "color": "RGB 38-111-160",
                "size": 16,
                "spacing": {"before": 12, "after": 2}
            },
            "Heading 3": {
                "font": "Lato Bold",
                "color": "RGB 0-8-14", 
                "size": 11,
                "spacing": {"before": 0, "after": 2}
            },
            "Heading 4": {
                "font": "Lato Italicized",
                "color": "RGB 0-8-14",
                "size": 11, 
                "spacing": {"before": 0, "after": 2}
            },
            "Normal (Body Text)": {
                "font": "Lato Regular",
                "color": "RGB 0-8-14",
                "size": 11,
                "spacing": {"before": 0, "after": 6}
            },
            "Captions/Callouts": {
                "font": "Lato Italicized", 
                "color": "RGB 0-8-14",
                "size": 10,
                "spacing": {"before": 0, "after": 6}
            },
            "Footers": {
                "font": "Lato Regular",
                "color": "RGB 0-8-14", 
                "size": 9,
                "spacing": {"before": 0, "after": 0}
            }
        }
    
    def get_rules_by_category(self, category: str) -> List[StyleRule]:
        """Get all rules for a specific category."""
        return [rule for rule in self.rules if rule.category == category]
    
    def get_critical_rules(self) -> List[StyleRule]:
        """Get all critical severity rules.""" 
        return [rule for rule in self.rules if rule.severity == "critical"]
    
    def get_pattern_rules(self) -> List[StyleRule]:
        """Get rules that have regex patterns for automated checking."""
        return [rule for rule in self.rules if rule.pattern is not None]

# DOCUMENT NAMING RULES
class AmidaDocumentNamingRules:
    """Document naming convention rules from ADMN-POL-1-11."""
    
    @staticmethod
    def get_naming_patterns() -> Dict[str, str]:
        """Standard Amida document naming patterns."""
        return {
            "amida_generated": r"^amida\s+.+\s+-[a-z]{3}\s+\d{1,2}\s+\d{4}\s+-[a-z]{2,3}$",
            "amida_final": r"^amida\s+.+\s+-[a-z]{3}\s+\d{4}\s+-final$", 
            "amida_draft": r"^draft\s+amida\s+.+\s+-[a-z]{3}\s+\d{1,2}\s+\d{4}\s+-[a-z]{2,3}$",
            "external_doc": r"^[a-z\s]+.+\s+-[a-z]{3}\s+\d{1,2}\s+\d{4}$",
        }
    
    @staticmethod
    def get_formatting_rules() -> List[str]:
        """Document title formatting rules."""
        return [
            "Use lowercase letters only in document titles",
            "Separate words with spaces, avoid dashes (-)", 
            "Use uppercase for proper names, acronyms, abbreviations (VA, DoD)",
            "Date format: [month] [day] [year] (e.g., apr 5 2025)",
            "Use lowercase initials for owner and editor"
        ]

# Initialize the rules instance
amida_rules = AmidaStyleRules()
naming_rules = AmidaDocumentNamingRules()