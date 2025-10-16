# Slide Review Agent

AI-powered presentation analysis tool that automatically reviews PowerPoint (PPTX) and PDF files for style compliance, tone, grammar, and risky content according to Amida's Style Guide. Built with FastAPI, spaCy, NLTK, and Azure OpenAI.

## Features

- **Multi-format Support**: Process both PPTX and PDF presentations with precise extraction
- **Comprehensive Style Compliance**: Enforces 50+ rules from Amida's Style Guide
- **AI-Powered Tone Analysis**: Detects negative language, passive voice, and hedging phrases
- **Advanced Grammar Checking**: Fixes numerals, punctuation, spacing, contractions, and word list preferences
- **Inclusivity Verification**: Ensures gender-neutral language and person-first terminology
- **Protection Layer**: Intelligent detection of protected content (names, dates, technical terms) to prevent false positives
- **LLM-Powered Suggestions**: Batch rewriting with semantic validation
- **Visual Reports**: Interactive web interface with severity breakdowns, category analysis, and slide navigation
- **Export Capabilities**: Download results as JSON or view detailed analysis reports

## Architecture

### Backend (FastAPI + Python)
- **Document Processing Pipeline**: Unified normalization for PPTX and PDF formats
- **Dual Analysis System**:
  - Rule-based grammar checker (50+ style rules)
  - LLM-powered tone analyzer (positive language, active voice)
- **Protection Layer**: LLM-based detection of protected content to avoid false positives
- **Provider-Agnostic LLM Client**: Supports Azure OpenAI, OpenAI, and other providers
- **Async Processing**: Non-blocking document analysis with caching

### Frontend (Vanilla HTML/CSS/JavaScript)
- Single-page application with real-time processing status
- Visual report mode with charts and statistics
- JSON output view for detailed inspection
- Responsive design with light/dark mode support
- No build step required

## Installation

### Prerequisites
- Python 3.9+
- Azure OpenAI API key (or compatible LLM provider)

### Setup

```bash
# Clone the repository
git clone https://github.com/shreya-bani/slide-review-agent.git
cd slide-review-agent

# Install dependencies
pip install -r requirements.txt

# Download required spaCy model
python -m spacy download en_core_web_sm

# Configure environment (copy and edit .env.example)
cp .env.example .env
# Edit .env with your LLM configuration
```

### Configuration

Edit `.env` file with your settings:

```env
# Server Configuration
HOST=127.0.0.1
PORT=8000
CORS_ORIGINS=*

# LLM Configuration (REQUIRED)
LLM_PROVIDER=azure
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-5-chat(your-model-here)
LLM_DEPLOY=gpt-5-chat(your-model-here)
LLM_API_ENDPOINT=https://your-endpoint.openai.azure.com
LLM_API_VERSION=2024-12-01-preview(your-version-here)

# Storage Directories
UPLOAD_DIR=./data/uploads
OUTPUT_DIR=./data/outputs
LOG_DIR=./data/logs
```

## Usage

### Start the Server

```bash
# Or using uvicorn directly
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### Access the Application

- **Frontend UI**: http://127.0.0.1:8000/app
- **API Root**: http://127.0.0.1:8000
- **Health Check**: http://127.0.0.1:8000/health

### Using the Web Interface

1. Navigate to http://127.0.0.1:8000/app
2. Drag and drop a PPTX or PDF file (max 50MB)
3. Wait for processing to complete
4. Review the visual report with:
   - Quality score and severity breakdown
   - Category analysis (grammar, tone, inclusivity)
   - Detailed findings table with suggestions
   - Slide-by-slide navigation
5. Export results or view raw JSON output

### API Endpoints

- `POST /upload-document`: Upload and analyze document
- `GET /health`: System health check (LLM, processors, analyzers)
- `GET /analysis-history`: Retrieve previous analysis metadata
- `GET /app`: Serve frontend application
