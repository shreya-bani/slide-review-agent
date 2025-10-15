FROM python:3.11-slim
# Set working directory
WORKDIR /app
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# Install system dependencies required for various Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements file from backend directory
COPY backend/requirements.txt ./requirements.txt
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Download spaCy language model
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl 
# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True)"
# Copy project files
COPY . .
# Create necessary directories
RUN mkdir -p data/uploads data/outputs data/logs
# Copy environment example to .env (will be overridden by volume mount or environment variables)
# RUN if [ -f .env.example ]; then cp .env.example .env; fi
# Expose port
EXPOSE 8000
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
# Run the application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]






