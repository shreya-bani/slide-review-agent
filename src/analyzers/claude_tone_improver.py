"""
Claude-powered tone improvement system
Provides intelligent, contextual suggestions for tone improvements
using Anthropic's Claude API
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import anthropic
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ToneImprovement:
    """Represents an intelligent tone improvement suggestion"""
    element_id: str
    original_text: str
    improved_text: str
    improvement_type: str  # 'active_voice', 'positive_language', 'overall_tone'
    explanation: str
    confidence: str  # 'high', 'medium', 'low'


class ClaudeToneImprover:
    """Uses Claude API to provide intelligent tone improvements"""
    
    def __init__(self):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Amida style guide context
        self.amida_context = """
        Amida's voice should be:
        - Technical but not condescending
        - Intelligent but not assuming  
        - Compelling but not hyperbolic
        - Conversational but not informal
        - Inquisitive but not indecisive
        - Passionate but not tangential
        
        Core values: Impact, Excellence, and Joy
        Always use active voice and positive language.
        """
    
    def improve_tone_issues(self, issues: List[Dict], elements: List[Dict], max_improvements: int = 10) -> List[ToneImprovement]:
        """
        Generate intelligent improvements for tone issues using Claude
        """
        improvements = []
        
        # Group issues by element for batch processing
        element_issues = {}
        for issue in issues:
            element_id = issue['element_id']
            if element_id not in element_issues:
                element_issues[element_id] = []
            element_issues[element_id].append(issue)
        
        # Get element text mapping
        element_texts = {elem['element_id']: elem['text'] for elem in elements}
        
        # Process up to max_improvements elements
        processed = 0
        for element_id, element_issues_list in element_issues.items():
            if processed >= max_improvements:
                break
                
            original_text = element_texts.get(element_id, '')
            if not original_text:
                continue
            
            try:
                improvement = self._get_claude_improvement(
                    original_text, element_issues_list, element_id
                )
                if improvement:
                    improvements.append(improvement)
                    processed += 1
                    
            except Exception as e:
                logger.error(f"Error getting Claude improvement for {element_id}: {e}")
                continue
        
        return improvements
    
    def _get_claude_improvement(self, original_text: str, issues: List[Dict], element_id: str) -> Optional[ToneImprovement]:
        """Get improvement suggestion from Claude for a specific text element"""
        
        # Prepare issue descriptions
        issue_descriptions = []
        for issue in issues:
            issue_descriptions.append(f"- {issue['issue_type']}: {issue['explanation']}")
        
        prompt = f"""You are helping improve business presentation content to match Amida's style guide.

{self.amida_context}

Original text that needs improvement:
"{original_text}"

Detected issues:
{chr(10).join(issue_descriptions)}

Please provide an improved version that:
1. Fixes the detected issues
2. Maintains the original meaning and technical accuracy
3. Follows Amida's voice guidelines
4. Uses active voice and positive language
5. Keeps the same level of technical detail

Provide your response in this exact format:
IMPROVED_TEXT: [your improved version]
EXPLANATION: [brief explanation of key changes made]
CONFIDENCE: [high/medium/low based on how certain you are this is better]"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-0",
                max_tokens=500,
                temperature=0.3,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            return self._parse_claude_response(response.content[0].text, original_text, element_id, issues)
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return None
    
    def _parse_claude_response(self, response_text: str, original_text: str, element_id: str, issues: List[Dict]) -> Optional[ToneImprovement]:
        """Parse Claude's response into a ToneImprovement object"""
        try:
            lines = response_text.strip().split('\n')
            improved_text = ""
            explanation = ""
            confidence = "medium"
            
            for line in lines:
                if line.startswith('IMPROVED_TEXT:'):
                    improved_text = line.replace('IMPROVED_TEXT:', '').strip()
                elif line.startswith('EXPLANATION:'):
                    explanation = line.replace('EXPLANATION:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    confidence = line.replace('CONFIDENCE:', '').strip().lower()
            
            if not improved_text or improved_text == original_text:
                return None
            
            # Determine primary improvement type
            improvement_type = "overall_tone"
            if any(issue['issue_type'] == 'passive_voice' for issue in issues):
                improvement_type = "active_voice"
            elif any(issue['issue_type'] == 'negative_language' for issue in issues):
                improvement_type = "positive_language"
            
            return ToneImprovement(
                element_id=element_id,
                original_text=original_text,
                improved_text=improved_text,
                improvement_type=improvement_type,
                explanation=explanation or "Improved tone and clarity",
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error parsing Claude response: {e}")
            return None
    
    def generate_overall_recommendations(self, tone_stats: Dict, improvements: List[ToneImprovement]) -> List[str]:
        """Generate intelligent overall recommendations using Claude"""
        
        prompt = f"""Based on this tone analysis of a business presentation, provide 3-5 specific, actionable recommendations:

Analysis Results:
- Total elements analyzed: {tone_stats['total_elements']}
- Average positivity score: {tone_stats['avg_positivity_score']:.2f}/1.0
- Passive voice issues: {tone_stats['passive_voice_count']}
- Negative language patterns: {tone_stats['negative_language_count']}
- Elements needing improvement: {tone_stats['elements_needing_llm_rewrite']}

{len(improvements)} specific improvements were generated.

Provide practical recommendations for the document author to improve the overall tone, following Amida's style guide (technical but not condescending, compelling but not hyperbolic, positive and active voice)."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-0",
                max_tokens=400,
                temperature=0.4,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            # Split response into individual recommendations
            recommendations = [
                line.strip().lstrip('•-123456789. ')
                for line in response.content[0].text.strip().split('\n')
                if line.strip() and not line.strip().startswith('Based on')
            ]
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [
                "Consider using more active voice throughout the document",
                "Focus on positive language to maintain an upbeat tone",
                "Review technical content for clarity without being condescending"
            ]


# Helper function
def get_claude_improvements(issues: List[Dict], elements: List[Dict], tone_stats: Dict) -> Dict[str, Any]:
    """Get intelligent tone improvements using Claude"""
    try:
        improver = ClaudeToneImprover()
        improvements = improver.improve_tone_issues(issues, elements, max_improvements=5)
        recommendations = improver.generate_overall_recommendations(tone_stats, improvements)
        
        return {
            'improvements': improvements,
            'recommendations': recommendations,
            'success': True
        }
    except Exception as e:
        logger.error(f"Error getting Claude improvements: {e}")
        return {
            'improvements': [],
            'recommendations': [
                f"Claude API error: {str(e)}",
                "Please check your ANTHROPIC_API_KEY in .env file"
            ],
            'success': False
        }