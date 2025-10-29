# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Slide Review Agent is an AI-powered presentation analysis tool that checks PowerPoint (PPTX) and PDF files for style compliance, tone, grammar, and risky content according to Amida's Style Guide. The system uses a FastAPI backend with Python processors and analyzers, and a vanilla HTML/CSS/JavaScript frontend.

## Running the Application

### Development Server
```bash
# Start the FastAPI backend server (runs on http://127.0.0.1:8000)
python -m backend.app

# Or using uvicorn directly
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### Access the Application
- Frontend UI: http://127.0.0.1:8000/app
- API root: http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/health

## Configuration

### Environment Setup
The application uses `.env.example` as the default configuration source (loaded with `override=False`). Key settings:

- **Server Configuration**:
  - `HOST`: Server host address (default: `127.0.0.1`)
  - `PORT`: Server port (default: `8000`)
  - `CORS_ORIGINS`: CORS allowed origins (default: `*`, can be comma-separated list)

- **LLM Configuration** (REQUIRED):
  - `LLM_PROVIDER`: Provider name (e.g., `huggingface`, `groq`)
  - `LLM_API_KEY`: API key for the LLM provider
  - `LLM_MODEL`: Model identifier (e.g., `google/gemma-2-2b-it`)
  - `LLM_API_ENDPOINT`: API endpoint URL

- **File Storage**:
  - `UPLOAD_DIR`: Where uploaded files are stored (default: `./data/uploads`)
  - `OUTPUT_DIR`: Where analysis results are saved (default: `./data/outputs`)
  - `LOGS_DIR`: Application logs (default: `./data/logs`)

### Testing Settings
```bash
# Verify configuration is valid
python -m backend.config.settings --verbose

# Test LLM connectivity
python -m backend.services.llm_health
```

## Architecture

### Document Processing Pipeline

The application follows a three-stage pipeline for document analysis:

1. **Document Upload & Normalization** (`backend/processors/`)
   - `enhanced_pptx_reader.py`: Extracts content from PowerPoint files using python-pptx
   - `pdf_reader.py`: Extracts content from PDF files
   - `document_normalizer.py`: Converts both formats to a unified JSON schema with standardized:
     - `UnifiedLocator`: Page/slide index, element type, position
     - `UnifiedFormatting`: Font details, styling, hierarchy
     - `UnifiedTextElement`: Text with locator and formatting metadata

2. **Style Analysis** (`backend/analyzers/`)
   - `simple_style_checker.py`: Rule-based grammar checking
     - Numeral formatting (spell out 1-9, use digits for 10+)
     - Spacing issues (double spaces, extra punctuation)
     - Quote standardization (smart quotes)
     - Word list compliance
     - Heading/title/bullet formatting
     - Protected content detection via LLM (names, dates, technical terms)

   - `advance_style_analyzer.py`: LLM-powered tone analysis with two main rules:
     - `positive_language`: VADER sentiment analysis to detect negative language (threshold: -0.05)
     - `active_voice`: Passive voice and hedging detection
     - Uses spaCy for NLP and batch LLM prompting for rewrite suggestions

   - `combine_analysis.py`: Orchestrates both analyzers and produces unified report

3. **Response Generation** (`backend/app.py`)
   - Combines analysis results with document metadata
   - Generates categorized issues (by severity, category, slide)
   - Produces statistics and summaries
   - Saves multiple output formats (normalized, analysis, result)

### LLM Integration

The application uses a provider-agnostic LLM client (`backend/utils/llm_client.py`):
- Single unified configuration via `settings.py`
- Simple `chat(messages)` interface
- Used for:
  - Protected content detection (avoiding false positives on names, dates, etc.)
  - Tone analysis and rewrite suggestions
  - Batch processing with masking of non-editable content

### File Naming Convention

Documents are assigned incremental IDs and cleaned filenames:
- Format: `{doc_id:03d}_{clean_stem}{extension}`
- Example: `001_Amida_Agentic_AI_solution_Strategic_Plan.pptx`
- Three output files per analysis:
  - `{file_id}_{clean_stem}_normalized.json`: Unified document structure
  - `{file_id}_{clean_stem}_combined_analysis.json`: Full analysis report
  - `{file_id}_{clean_stem}_result.json`: Frontend-optimized response

## Code Organization

```
backend/
├── app.py                    # FastAPI application & document processing pipeline
├── config/
│   ├── settings.py          # Centralized configuration using pydantic-settings
│   └── style_rules.py       # Style guide rules and patterns
├── processors/
│   ├── enhanced_pptx_reader.py       # PowerPoint content extraction
│   ├── pdf_reader.py        # PDF content extraction
│   └── document_normalizer.py # Format unification
├── analyzers/
│   ├── simple_style_checker.py    # Grammar and formatting rules
│   ├── advance_style_analyzer.py  # Tone and voice analysis
│   └── combine_analysis.py        # Analysis orchestration
├── utils/
│   ├── llm_client.py        # LLM communication layer
│   └── llm_health.py        # LLM connectivity checks
└── services/               # Additional extraction utilities

frontend/
├── pages/main.html        # Main application interface
├── scripts/script.js       # Frontend logic and API calls
└── styles/main.css       # Application styling

data/
├── uploads/                # Uploaded documents (auto-created)
├── outputs/                # Analysis results (auto-created)
└── logs/                   # Application logs (auto-created)
```

## Key Design Patterns

### Async Processing
- Uses `asyncio.to_thread()` for CPU-bound operations (document normalization, analysis)
- Maintains async FastAPI handlers for I/O operations
- Processing is cached by `file_id` to avoid re-analysis

### Error Handling
- Processors return `success: false` with error details on failure
- LLM failures fall back to rule-based detection where possible
- Missing configuration validated via `settings.validate_llm_config()`

### Dataclass-based Models
- Uses Python `@dataclass` for structured data (UnifiedLocator, StyleIssue, etc.)
- Provides `to_dict()` methods for JSON serialization
- Type hints throughout for clarity

## Analyzer-Specific Details

### Protected Content Strategy
The `ComprehensiveProtectionDetector` in `simple_style_checker.py` uses a single upfront LLM call to identify:
- Protected names (people, companies, organizations)
- Technical terms and abbreviations
- Dates and numbers that should not be modified
- IDs and reference numbers

This prevents false positives in grammar/formatting rules.

### Tone Analysis Configuration
`advance_style_analyzer.py` uses configurable thresholds:
- `VADER_NEG_THRESHOLD`: -0.05 (base negativity threshold)
- `VADER_HARD_NEG`: -0.40 (stricter threshold for sentiment-only flags)
- Skips SWOT analysis sections and heading-only elements
- Uses domain-specific whitelists to avoid false negatives

### Batch Rewriting
LLM rewrite suggestions are batched (max 50 items) with:
- Content masking for protected elements
- Constraint extraction (preserve numbers, dates, names)
- Output validation and rejection logging
- Fallback to original text if LLM suggestion is invalid

## Common Development Tasks

### Adding a New Style Rule
1. Define rule pattern in `backend/config/style_rules.py`
2. Implement detection logic in `simple_style_checker.py` (grammar) or `advance_style_analyzer.py` (tone)
3. Create `StyleIssue` objects with appropriate severity and category
4. Update rule documentation in docstrings

### Modifying LLM Prompts
- Grammar/protection prompts: `simple_style_checker.py` → `_llm_comprehensive_detection()`
- Tone/rewrite prompts: `advance_style_analyzer.py` → `_batch_rewrite_llm()`
- Always include output format specifications and examples in prompts
- Test with `--verbose` flag for detailed logging

### Testing Document Processing
```bash
# Test normalization alone
python -m backend.processors.document_normalizer

# Test combined analysis
python -m backend.analyzers.combine_analysis <normalized.json> [output.json]

# Test with verbose logging
python -m backend.analyzers.advance_style_analyzer <normalized.json> --no-rewrite -v
```

## Important Implementation Notes

### Windows Path Handling
This codebase runs on Windows (`win32` platform). Use `Path` objects from `pathlib` for cross-platform compatibility and avoid hardcoded path separators.

### Git Workflow
- Main branch: `main`
- Current working branch: `version-2`
- Many output files and test data in `data/` are git-ignored (see git status)

### LLM Rate Limiting
The application may implement rate limiting via:
- `REWRITE_BUDGET_MAX_CALLS`: Maximum LLM calls per window
- `REWRITE_BUDGET_WINDOW_HOURS`: Time window for rate limiting
- Check settings if LLM calls are unexpectedly disabled

### Frontend Architecture
The frontend is intentionally simple (vanilla JS, no framework):
- Single-page application in `frontend/pages/main.html`
- Direct API calls via `fetch()` in `frontend/scripts/script.js`
- Displays analysis results in tables with slide navigation
- No build step required
