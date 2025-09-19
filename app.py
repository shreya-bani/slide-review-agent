"""
Slide Review Agent - Complete Input Handling
Following exact specifications for PPTX and PDF processing
"""

import streamlit as st
from src.processors import process_document


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
                        
                        # Display processing summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Slides/Pages", result['total_slides'])
                        with col2:
                            st.metric("Text Elements", result['metadata']['total_elements'])
                        with col3:
                            st.metric("Document Type", result['document_type'].upper())
                        
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
                        st.write("4. ⏳ Tone checking (Active Voice + Positive Language)")
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
        st.write("- Style, location, size attributes")

        st.header("🎯 Amida Core Values")
        st.write("- **Impact**")
        st.write("- **Excellence**")
        st.write("- **Joy**")


if __name__ == "__main__":
    main()