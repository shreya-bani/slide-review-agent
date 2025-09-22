"""
Slide Review Agent - Complete Input Handling + Tone + Usage Analysis
Following exact specifications for PPTX and PDF processing with tone/usage checking
Enhanced to show detailed detections with page/line references
"""

import os
import warnings
import streamlit as st

from src.processors import process_document
from src.analyzers import analyze_tone, get_groq_improvements, analyze_usage

warnings.filterwarnings("ignore", message=".*non-stroke color.*")


def main():
    st.title("🔍 Slide Review Agent")
    st.subheader("Amida Style Guide Compliance Checker")

    # File upload - both PPTX and PDF
    uploaded_file = st.file_uploader(
        "Upload your presentation",
        type=["pptx", "pdf"],
        help="Supports PowerPoint files (.pptx) and PDF files (.pdf)",
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
                    file_content = uploaded_file.read()
                    result = process_document(file_content, uploaded_file.name)

                    if result["processing_status"] != "success":
                        st.error(f"❌ Failed to process document: {result.get('error_message', 'Unknown error')}")
                        return

                    st.success(
                        f"✅ Successfully processed {result['document_type'].upper()} with {result['total_slides']} slides/pages!"
                    )

                    # -----------------------
                    # Tone analysis
                    # -----------------------
                    with st.spinner("Analyzing tone and language patterns..."):
                        tone_analysis = analyze_tone(result["elements"])

                    # Groq-powered tone improvements (optional)
                    groq_improvements = None
                    improved_by_elem = {}
                    if os.getenv("GROQ_API_KEY"):
                        if tone_analysis["issues"]:
                            with st.spinner("Getting intelligent suggestions from Groq..."):
                                groq_improvements = get_groq_improvements(
                                    tone_analysis["issues"],
                                    result["elements"],
                                    tone_analysis["overall_stats"],
                                )
                            if groq_improvements and groq_improvements.get("success"):
                                improvements = groq_improvements.get("improvements", [])
                                improved_by_elem = {imp.element_id: imp for imp in improvements}
                    else:
                        st.info("💡 Add GROQ_API_KEY to .env for intelligent tone improvements")

                    # -----------------------
                    # Usage analysis (Specificity + Inclusivity)
                    # -----------------------
                    with st.spinner("Checking usage: specificity and inclusivity..."):
                        usage_result = analyze_usage(
                            result["elements"], rules_path="src/rules/amida_style_rules.json"
                        )
                    usage_stats = usage_result["overall_stats"]

                    # -----------------------
                    # Processing summary
                    # -----------------------
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Slides/Pages", result["total_slides"])
                    with col2:
                        st.metric("Text Elements", result["metadata"]["total_elements"])
                    with col3:
                        st.metric("Passive Voice", tone_analysis["overall_stats"]["passive_voice_count"])
                    with col4:
                        st.metric("Negative Language", tone_analysis["overall_stats"]["negative_language_count"])
                    with col5:
                        st.metric("Avg Positivity", f"{tone_analysis['overall_stats']['avg_positivity_score']:.2f}")

                    # -----------------------
                    # Tone Analysis UI
                    # -----------------------
                    st.write("## 🎯 Tone Analysis Results")

                    tone_stats = tone_analysis["overall_stats"]
                    if tone_stats["avg_positivity_score"] >= 0.7:
                        st.success(f"✅ Good overall positivity score: {tone_stats['avg_positivity_score']:.2f}/1.0")
                    else:
                        st.warning(
                            f"⚠️ Low positivity score: {tone_stats['avg_positivity_score']:.2f}/1.0 (threshold: 0.7)"
                        )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Passive Voice Issues", tone_stats["passive_voice_count"])
                    with col2:
                        st.metric("Negative Language Issues", tone_stats["negative_language_count"])
                    with col3:
                        st.metric("Low Positivity Elements", tone_stats.get("low_positivity_count", 0))

                    if tone_analysis["issues"]:
                        st.write("### 🔍 Tone Issues & Intelligent Suggestions")

                        passive_voice_issues = [
                            issue for issue in tone_analysis["issues"] if issue["issue_type"] == "passive_voice"
                        ]
                        negative_language_issues = [
                            issue for issue in tone_analysis["issues"] if issue["issue_type"] == "negative_language"
                        ]
                        low_positivity_issues = [
                            issue for issue in tone_analysis["issues"] if issue["issue_type"] == "low_positivity"
                        ]

                        # Groq-powered improvements (tone)
                        if groq_improvements and groq_improvements["success"] and groq_improvements["improvements"]:
                            st.write("#### ✨ Groq-Powered Improvements")
                            for improvement in groq_improvements["improvements"]:
                                confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                                confidence_icon = confidence_color.get(improvement.confidence, "🟡")

                                with st.expander(
                                    f"{confidence_icon} {improvement.improvement_type.replace('_', ' ').title()} - {improvement.element_id}",
                                    expanded=True,
                                ):
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.write("**Original:**")
                                        st.write(f"*{improvement.original_text}*")
                                    with c2:
                                        st.write("**Improved:**")
                                        st.success(improvement.improved_text)
                                    st.write(f"**Explanation:** {improvement.explanation}")
                                    st.write(f"**Confidence:** {improvement.confidence.title()}")
                                    ca, cb, cc = st.columns(3)
                                    with ca:
                                        st.button("✅ Approve", key=f"approve_{improvement.element_id}")
                                    with cb:
                                        st.button("✏️ Edit", key=f"edit_{improvement.element_id}")
                                    with cc:
                                        st.button("❌ Reject", key=f"reject_{improvement.element_id}")

                        # Passive voice issues
                        if passive_voice_issues:
                            st.write("#### 🔴 Passive Voice Issues")
                            for issue in passive_voice_issues:
                                page_ref = issue.get("page_line_ref", issue["element_id"])
                                with st.expander(f"🔴 Passive Voice - {page_ref}", expanded=False):
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.write("**Original:**")
                                        st.write(f"*{issue['original_text']}*")
                                    with c2:
                                        st.write("**Suggested Fix:**")
                                        imp = improved_by_elem.get(issue["element_id"])
                                        if imp and getattr(imp, "improved_text", None):
                                            st.success(imp.improved_text)
                                        else:
                                            st.info(issue["suggested_fix"])
                                    st.write(f"**Location:** {page_ref}")
                                    st.write(f"**Explanation:** {issue['explanation']}")

                        # Negative language issues
                        if negative_language_issues:
                            st.write("#### 🟡 Negative Language Issues")
                            for issue in negative_language_issues:
                                page_ref = issue.get("page_line_ref", issue["element_id"])
                                with st.expander(f"🟡 Negative Language - {page_ref}", expanded=False):
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.write("**Negative Word/Phrase:**")
                                        st.write(f"*{issue['original_text']}*")
                                    with c2:
                                        st.write("**Positive Alternative:**")
                                        st.success(issue["suggested_fix"])
                                    st.write(f"**Location:** {page_ref}")
                                    st.write(f"**Explanation:** {issue['explanation']}")

                        # Remaining tone issues
                        remaining_issues = [
                            issue
                            for issue in tone_analysis["issues"]
                            if (
                                groq_improvements is None
                                or not any(
                                    imp.element_id == issue["element_id"]
                                    for imp in groq_improvements.get("improvements", [])
                                )
                            )
                            and issue["issue_type"] not in ["passive_voice", "negative_language"]
                        ]
                        if remaining_issues:
                            st.write("#### 📋 Additional Issues Requiring LLM Review")
                            for issue in remaining_issues[:10]:
                                page_ref = issue.get("page_line_ref", issue["element_id"])
                                issue_type_color = "🔵" if issue["issue_type"] == "low_positivity" else "🟡"
                                with st.expander(
                                    f"{issue_type_color} {issue['issue_type'].replace('_', ' ').title()} - {page_ref}",
                                    expanded=False,
                                ):
                                    st.write("**Text:**")
                                    st.write(f"*{issue['original_text']}*")
                                    st.write(f"**Issue:** {issue['explanation']}")
                                    st.write(f"**Location:** {page_ref}")

                    # -----------------------
                    # Usage Analysis UI
                    # -----------------------
                    st.write("## 🧭 Usage Check (Specificity & Inclusivity)")
                    u1, u2, u3, u4, u5 = st.columns(5)
                    with u1:
                        st.metric("Vague Terms", usage_stats["specificity_vague"])
                    with u2:
                        st.metric("Temporal Words", usage_stats["specificity_temporal"])
                    with u3:
                        st.metric("Missing 4W+1H", usage_stats["specificity_missing_4w1h"])
                    with u4:
                        st.metric("Gendered Terms", usage_stats["inclusivity_gendered"])
                    with u5:
                        st.metric("Pronoun Issues", usage_stats["inclusivity_pronoun"])
                    st.caption(f"Slide coherence (↑ better): {usage_stats['avg_tangent_coherence']:.2f}")

                    if usage_result["issues"]:
                        st.write("### 🔎 Usage Issues & Suggested Rewrites")
                        for issue in usage_result["issues"]:
                            badge = "🟥" if issue["issue_type"] == "inclusivity" else "🟦"
                            title = f"{badge} {issue['issue_type'].title()} – {issue['subtype'].replace('_',' ').title()} – {issue.get('page_line_ref', issue['element_id'])}"
                            with st.expander(title, expanded=False):
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.write("**Original:**")
                                    st.write(f"*{issue['original_text']}*")
                                with c2:
                                    st.write("**Suggestion:**")
                                    st.success(issue["suggested_fix"])
                                st.write(f"**Explanation:** {issue['explanation']}")
                                st.write(f"**Confidence:** {issue['confidence']:.2f}")
                                b1, b2, b3 = st.columns(3)
                                with b1:
                                    st.button("✅ Approve", key=f"u_ok_{issue['element_id']}_{issue['subtype']}")
                                with b2:
                                    st.button("✏️ Edit", key=f"u_ed_{issue['element_id']}_{issue['subtype']}")
                                with b3:
                                    st.button("❌ Reject", key=f"u_no_{issue['element_id']}_{issue['subtype']}")
                    else:
                        st.success("No usage issues detected.")

                    # -----------------------
                    # Structured content UI
                    # -----------------------
                    st.write("## 📊 Structured Document Analysis")

                    slides_data = {}
                    for element in result["elements"]:
                        slides_data.setdefault(element["slide_number"], []).append(element)

                    for slide_num in sorted(slides_data.keys()):
                        elements = slides_data[slide_num]
                        slide_title = next(
                            (
                                e["text"][:50] + "..." if len(e["text"]) > 50 else e["text"]
                                for e in elements
                                if e["element_type"] == "title"
                            ),
                            f"Slide {slide_num}",
                        )

                        with st.expander(f"📄 {result['document_type'].upper()} Page {slide_num}: {slide_title}", expanded=False):
                            for element in elements:
                                st.write(f"**{element['element_type'].title()} Element** (`{element['element_id']}`)")

                                if element["element_type"] == "bullet":
                                    st.write("```")
                                    st.write(element["text"])
                                    st.write("```")
                                else:
                                    st.write(f"*{element['text']}*")

                                if element.get("style") or element.get("location") or element.get("font_info"):
                                    mc1, mc2, mc3 = st.columns(3)
                                    with mc1:
                                        if element.get("style"):
                                            st.write("**Style:**")
                                            for key, value in element["style"].items():
                                                if value is not None:
                                                    st.write(f"• {key}: {value}")
                                    with mc2:
                                        if element.get("location"):
                                            st.write("**Location:**")
                                            for key, value in element["location"].items():
                                                st.write(f"• {key}: {value}")
                                    with mc3:
                                        if element.get("font_info"):
                                            st.write("**Font:**")
                                            for key, value in element["font_info"].items():
                                                if value is not None:
                                                    st.write(f"• {key}: {value}")
                                st.write("---")

                    with st.expander("🔍 Raw JSON Structure", expanded=False):
                        st.json(result)

                    # -----------------------
                    # Progress
                    # -----------------------
                    st.write("## 🚀 Implementation Progress")
                    st.write("1. ✅ Complete input handling (PPTX + PDF)")
                    st.write("2. ✅ Structured JSON with metadata (style, location, size)")
                    st.write("3. ✅ Element classification (title, bullet, body, note)")
                    st.write("4. ✅ Enhanced tone checking (Active Voice + Positive Language + Page References)")
                    st.write("5. ✅ Usage checking (Specificity + Inclusivity)")
                    st.write("6. ⏳ Grammar & Style Rules enforcement")

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This tool checks presentation slides against Amida's Style Guide.")

        st.header("📋 Current Features")
        st.write("- File upload (PPTX + PDF)")
        st.write("- Complete metadata extraction")
        st.write("- Structured JSON output")
        st.write("- Element classification")
        st.write("- Active voice detection (spaCy)")
        st.write("- Enhanced negative language detection")
        st.write("- Positive language scoring (VADER)")
        st.write("- Page/line reference tracking")
        st.write("- Groq-powered intelligent improvements")
        st.write("- Usage checks (Specificity + Inclusivity)")

        st.header("🎯 Amida Core Values")
        st.write("- **Impact**")
        st.write("- **Excellence**")
        st.write("- **Joy**")


if __name__ == "__main__":
    main()
