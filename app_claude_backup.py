"""
Slide Review Agent - Complete Input Handling + Tone Analysis
Following exact specifications for PPTX and PDF processing with tone checking
"""

import streamlit as st
from src.processors import process_document
from src.analyzers import analyze_tone, get_claude_improvements
import os


def main():
    st.title("🔍 Slide Review Agent")
    st.subheader("Amida Style Guide Compliance Checker")

    # File upload - both PPTX and PDF
    uploaded_file = st.file_uploader(
        "Upload your presentation",
        type=['pptx', 'pdf'],
        help="Supports PowerPoint files (.pptx) and PDF files (.pdf)"
    )

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

        # Display file info
        st.write("**File Details:**")
        st.write(f"- Name: {uploaded_file.name}")
        st.write(f"- Size: {uploaded_file.size} bytes")
        st.write(f"- Type: {uploaded_file.type}")

        # Process button
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing document with complete metadata extraction..."):
                try:
                    # Process document with structured JSON output
                    file_content = uploaded_file.read()
                    result = process_document(file_content, uploaded_file.name)
                    
                    if result["processing_status"] == "success":
                        st.success(f"✅ Successfully processed {result['document_type'].upper()} with {result['total_slides']} slides/pages!")
                        
                        # Perform tone analysis
                        with st.spinner("Analyzing tone and language patterns..."):
                            tone_analysis = analyze_tone(result["elements"])
                        
                        # Get Claude-powered improvements if API key is available
                        claude_improvements = None
                        if os.getenv('ANTHROPIC_API_KEY'):
                            if tone_analysis['issues']:
                                with st.spinner("Getting intelligent suggestions from Claude..."):
                                    claude_improvements = get_claude_improvements(
                                        tone_analysis['issues'], 
                                        result["elements"], 
                                        tone_analysis['overall_stats']
                                    )
                        else:
                            st.info("💡 Add ANTHROPIC_API_KEY to .env for intelligent tone improvements")
                        
                        # Display processing summary
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Slides/Pages", result['total_slides'])
                        with col2:
                            st.metric("Text Elements", result['metadata']['total_elements'])
                        with col3:
                            st.metric("Tone Issues", len(tone_analysis['issues']))
                        with col4:
                            st.metric("Avg Positivity", f"{tone_analysis['overall_stats']['avg_positivity_score']:.2f}")
                        
                        # Display tone analysis summary
                        st.write("## 🎯 Tone Analysis Results")
                        
                        # Tone statistics
                        tone_stats = tone_analysis['overall_stats']
                        
                        if tone_stats['avg_positivity_score'] >= 0.7:
                            st.success(f"✅ Good overall positivity score: {tone_stats['avg_positivity_score']:.2f}/1.0")
                        else:
                            st.warning(f"⚠️ Low positivity score: {tone_stats['avg_positivity_score']:.2f}/1.0 (threshold: 0.7)")
                        
                        # Issues breakdown
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Passive Voice Issues", tone_stats['passive_voice_count'])
                        with col2:
                            st.metric("Negative Language", tone_stats['negative_language_count'])
                        with col3:
                            st.metric("Need LLM Rewrite", tone_stats['elements_needing_llm_rewrite'])
                        
                        # Display specific issues and improvements
                        if tone_analysis['issues']:
                            st.write("### 🔍 Tone Issues & Intelligent Suggestions")
                            
                            # Show Claude improvements if available
                            if claude_improvements and claude_improvements['success'] and claude_improvements['improvements']:
                                st.write("#### ✨ Claude-Powered Improvements")
                                
                                for improvement in claude_improvements['improvements']:
                                    confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                                    confidence_icon = confidence_color.get(improvement.confidence, "🟡")
                                    
                                    with st.expander(f"{confidence_icon} {improvement.improvement_type.replace('_', ' ').title()} - {improvement.element_id}", expanded=True):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**Original:**")
                                            st.write(f"*{improvement.original_text}*")
                                        
                                        with col2:
                                            st.write("**Improved:**")
                                            st.success(improvement.improved_text)
                                        
                                        st.write(f"**Explanation:** {improvement.explanation}")
                                        st.write(f"**Confidence:** {improvement.confidence.title()}")
                                        
                                        # Add approve/reject buttons (placeholder for future reviewer system)
                                        col_a, col_b, col_c = st.columns(3)
                                        with col_a:
                                            st.button("✅ Approve", key=f"approve_{improvement.element_id}", help="Accept this improvement")
                                        with col_b:
                                            st.button("✏️ Edit", key=f"edit_{improvement.element_id}", help="Modify this suggestion")
                                        with col_c:
                                            st.button("❌ Reject", key=f"reject_{improvement.element_id}", help="Decline this improvement")
                            
                            # Show remaining basic issues
                            basic_issues = [issue for issue in tone_analysis['issues'] 
                                          if claude_improvements is None or not any(imp.element_id == issue['element_id'] 
                                                                                   for imp in claude_improvements.get('improvements', []))]
                            
                            if basic_issues:
                                st.write("#### 📋 Additional Detected Issues")
                                for issue in basic_issues[:5]:  # Limit to prevent overwhelming
                                    issue_type_color = "🔴" if issue['issue_type'] == 'passive_voice' else "🟡"
                                    
                                    with st.expander(f"{issue_type_color} {issue['issue_type'].replace('_', ' ').title()}", expanded=False):
                                        st.write("**Text:**")
                                        st.write(f"*{issue['original_text']}*")
                                        st.write(f"**Basic Suggestion:** {issue['suggested_fix']}")
                                        st.write(f"**Element:** {issue['element_id']}")
                        
                        # Intelligent recommendations
                        st.write("### 💡 Recommendations")
                        if claude_improvements and claude_improvements['success']:
                            for recommendation in claude_improvements['recommendations']:
                                st.info(recommendation)
                        else:
                            # Fallback to basic recommendations
                            for recommendation in tone_analysis['recommendations']:
                                st.info(recommendation)
                        
                        # Display structured content
                        st.write("## 📊 Structured Document Analysis")
                        
                        # Group elements by slide
                        slides_data = {}
                        for element in result["elements"]:
                            slide_num = element["slide_number"]
                            if slide_num not in slides_data:
                                slides_data[slide_num] = []
                            slides_data[slide_num].append(element)
                        
                        # Display each slide with detailed metadata
                        for slide_num in sorted(slides_data.keys()):
                            elements = slides_data[slide_num]
                            slide_title = next((e["text"][:50] + "..." if len(e["text"]) > 50 else e["text"] 
                                              for e in elements if e["element_type"] == "title"), f"Slide {slide_num}")
                            
                            with st.expander(f"📄 {result['document_type'].upper()} Page {slide_num}: {slide_title}", expanded=False):
                                for element in elements:
                                    st.write(f"**{element['element_type'].title()} Element** (`{element['element_id']}`)")
                                    
                                    # Display text content
                                    if element['element_type'] == 'bullet':
                                        st.write("```")
                                        st.write(element['text'])
                                        st.write("```")
                                    else:
                                        st.write(f"*{element['text']}*")
                                    
                                    # Display metadata in columns
                                    if element.get('style') or element.get('location') or element.get('font_info'):
                                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                                        
                                        with meta_col1:
                                            if element.get('style'):
                                                st.write("**Style:**")
                                                for key, value in element['style'].items():
                                                    if value is not None:
                                                        st.write(f"• {key}: {value}")
                                        
                                        with meta_col2:
                                            if element.get('location'):
                                                st.write("**Location:**")
                                                for key, value in element['location'].items():
                                                    st.write(f"• {key}: {value}")
                                        
                                        with meta_col3:
                                            if element.get('font_info'):
                                                st.write("**Font:**")
                                                for key, value in element['font_info'].items():
                                                    if value is not None:
                                                        st.write(f"• {key}: {value}")
                                    
                                    st.write("---")
                        
                        # Show JSON structure (collapsed)
                        with st.expander("🔍 Raw JSON Structure", expanded=False):
                            st.json(result)
                        
                        # Show next steps
                        st.write("## 🚀 Implementation Progress")
                        st.write("1. ✅ Complete input handling (PPTX + PDF)")
                        st.write("2. ✅ Structured JSON with metadata (style, location, size)")
                        st.write("3. ✅ Element classification (title, bullet, body, note)")
                        st.write("4. ✅ Tone checking (Active Voice + Positive Language)")
                        st.write("5. ⏳ Usage checking (Specificity + Inclusivity)")
                        st.write("6. ⏳ Grammar & Style Rules enforcement")
                    
                    else:
                        st.error(f"❌ Failed to process document: {result.get('error_message', 'Unknown error')}")

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This tool checks presentation slides against Amida's Style Guide.")

        st.header("📋 Current Features")
        st.write("- File upload (PPTX + PDF)")
        st.write("- Complete metadata extraction")
        st.write("- Structured JSON output")
        st.write("- Element classification")
        st.write("- Active voice detection (spaCy)")
        st.write("- Positive language scoring (VADER)")
        st.write("- Tone issue identification")

        st.header("🎯 Amida Core Values")
        st.write("- **Impact**")
        st.write("- **Excellence**")
        st.write("- **Joy**")


if __name__ == "__main__":
    main()