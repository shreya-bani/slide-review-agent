# Slide Review Agent

AI-powered editor and reviewer agent for presentation slides. Automatically checks tone, grammar, style compliance, and flags risky content in PPTX/PDF files according to Amida's Style Guide. Built with Python, spaCy, NLTK, and Streamlit.

## Features

- **Multi-format Support**: Process both PPTX and PDF presentations
- **Style Compliance**: Enforces Amida's comprehensive style guide
- **Tone Analysis**: Checks for active voice and positive language
- **Grammar & Mechanics**: Fixes common grammar issues, punctuation, and formatting
- **Inclusivity Check**: Ensures gender-neutral and inclusive language
- **Risk Detection**: Flags potentially sensitive content (numbers, dates, client names)
- **Interactive Review**: Streamlit interface for human review and approval

## Installation

```bash
git clone https://github.com/shreya-bani/slide-review-agent.git
cd slide-review-agent
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```bash
streamlit run src/ui/app.py
```
