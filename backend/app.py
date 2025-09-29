"""
FastAPI application entry point for Slide Review Agent.
"""
import asyncio
import os
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from .config.settings import settings

# Configure the basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import document processors with fallback handling
try:
    from .processors.pptx_reader import PPTXReader
    from .processors.pdf_reader import PDFReader
    from .processors.document_normalizer import DocumentNormalizer
except Exception as e:
    logger.warning(f"Could not import document processors: {e}")
    PPTXReader = None
    PDFReader = None
    DocumentNormalizer = None

# Global variable to track if processing has already been done
processed_files = {}

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered slide deck review agent",
        version="1.0.0",
        debug=settings.debug
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Ensure required directories exist
    settings.ensure_directories()
    return app

app = create_app()

# Frontend static file paths
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
STYLES_DIR = FRONTEND_DIR / "styles"
SCRIPTS_DIR = FRONTEND_DIR / "scripts"
PAGES_DIR = FRONTEND_DIR / "pages"

# Mount static files
if not all([STYLES_DIR.exists(), SCRIPTS_DIR.exists(), FRONTEND_DIR.exists()]):
    logger.warning(f"Frontend directories not found under {FRONTEND_DIR}")

app.mount("/styles", StaticFiles(directory=str(STYLES_DIR)), name="styles")
app.mount("/scripts", StaticFiles(directory=str(SCRIPTS_DIR)), name="scripts")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Helper functions for file naming
def get_next_document_id() -> int:
    """Get the next incremental document ID by checking existing files."""
    output_dir = Path(settings.output_dir)
    if not output_dir.exists():
        return 1
    
    existing_ids = []
    for file_path in output_dir.glob("*_analysis.json"):
        try:
            filename = file_path.stem
            # Extract the first part before the first underscore
            parts = filename.split('_')
            if parts and parts[0].isdigit():
                existing_ids.append(int(parts[0]))
        except:
            continue
    
    return max(existing_ids, default=0) + 1

def create_clean_filename(original_filename: str, doc_id: int) -> tuple[str, str]:
    """Create clean filename and file_id from original filename and incremental ID."""
    stem = Path(original_filename).stem
    extension = Path(original_filename).suffix.lower()
    
    # Clean filename: remove problematic characters, limit length
    clean_stem = "".join(c for c in stem if c.isalnum() or c in " -_").strip()
    clean_stem = clean_stem.replace(" ", "_")
    
    if len(clean_stem) > 50:
        clean_stem = clean_stem[:50]
    
    file_id = f"{doc_id:03d}"  # 001, 002, 003, etc.
    clean_filename = f"{file_id}_{clean_stem}{extension}"
    
    return clean_filename, file_id

def write_json(path: Path, obj: dict) -> None:
    """Helper to write JSON files with proper formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "environment": settings.environment,
        "demo_mode": settings.demo_mode
    }

@app.get("/app")
async def serve_frontend():
    """Serve the frontend application."""
    index_file = PAGES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail=f"Frontend index not found: {index_file}")
    return FileResponse(str(index_file))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database path
        db_path = Path(settings.database_path).parent
        db_accessible = db_path.exists() or db_path.parent.exists()
        
        # Test directories
        dirs_exist = (
            Path(settings.upload_dir).exists() and 
            Path(settings.output_dir).exists()
        )
        
        # Test LLM configuration
        llm_config_valid = settings.validate_llm_config()
        
        # Test document processors
        processors_available = all([PPTXReader, PDFReader, DocumentNormalizer])
        
        health_status = {
            "database": "ok" if db_accessible else "error",
            "directories": "ok" if dirs_exist else "error", 
            "llm_config": "ok" if llm_config_valid else "error",
            "processors": "ok" if processors_available else "error",
            "overall": "healthy" if all([db_accessible, dirs_exist, llm_config_valid, processors_available]) else "unhealthy"
        }
        
        status_code = 200 if health_status["overall"] == "healthy" else 503
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "overall": "unhealthy"}, 
            status_code=503
        )

@app.get("/config")
async def get_config():
    """Get non-sensitive configuration info."""
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.groq_model if settings.llm_provider == "groq" else settings.openai_model,
        "max_file_size_mb": settings.max_file_size_mb,
        "demo_mode": settings.demo_mode,
        "has_groq_key": bool(settings.groq_api_key),
        "has_openai_key": bool(settings.openai_api_key),
        "allowed_file_types": [".pptx", ".pdf"],
        "upload_dir": settings.upload_dir,
        "processors_status": {
            "pptx_reader": "available" if PPTXReader else "missing",
            "pdf_reader": "available" if PDFReader else "missing",
            "document_normalizer": "available" if DocumentNormalizer else "missing"
        }
    }

@app.get("/debug-env")
async def debug_env():
    """Debug environment configuration."""
    return {
        "groq_api_key_exists": bool(settings.groq_api_key),
        "groq_api_key_length": len(settings.groq_api_key) if settings.groq_api_key else 0,
        "groq_api_key_first_4": settings.groq_api_key[:4] if settings.groq_api_key else None,
        "llm_provider": settings.llm_provider,
        "groq_model": settings.groq_model,
        "current_working_dir": str(Path.cwd()),
        "processors_available": {
            "pptx_reader": PPTXReader is not None,
            "pdf_reader": PDFReader is not None,
            "document_normalizer": DocumentNormalizer is not None
        }
    }

@app.post("/upload-document")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_info: Optional[str] = Form(None)
):
    """Upload and process a document."""
    user_info = user_info or "Anonymous"

    # Validate file extension
    allowed_extensions = ['.pptx', '.pdf']
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type. Allowed: {allowed_extensions}")

    # Check file size
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    cl = request.headers.get("content-length")
    if cl and int(cl) > max_bytes:
        raise HTTPException(400, f"File too large. Maximum size: {settings.max_file_size_mb}MB")

    # Generate clean filename and ID
    doc_id = get_next_document_id()
    clean_filename, file_id = create_clean_filename(file.filename, doc_id)
    file_path = Path(settings.upload_dir) / clean_filename

    logger.info(f"Processing document {doc_id}: {file.filename} -> {clean_filename}")

    # Stream file to disk with size enforcement
    written = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    buffer.close()
                    try: 
                        os.remove(file_path)
                    except: 
                        pass
                    raise HTTPException(400, f"File too large. Maximum size: {settings.max_file_size_mb}MB")
                buffer.write(chunk)
    except Exception as e:
        try: 
            os.remove(file_path)
        except: 
            pass
        raise HTTPException(500, f"Failed to save uploaded file: {str(e)}")

    # Check if this file has already been processed
    if file_id in processed_files:
        logger.info(f"Document {file_id} already processed, returning existing result")
        return JSONResponse(content=processed_files[file_id], status_code=200)

    # Process the document ONLY ONCE
    processing_result = await process_document_async(
        file_path=str(file_path),
        original_filename=file.filename,
        user_info=user_info,
        file_id=file_id,
        doc_id=doc_id
    )
    
    # Cache the result to prevent duplicate processing
    processed_files[file_id] = processing_result
    
    return JSONResponse(content=processing_result, status_code=200)

async def process_document_async(file_path: str, original_filename: str,
                                 user_info: str, file_id: str, doc_id: int) -> dict:
    """Process a document asynchronously and save all JSON artifacts."""
    try:
        if not DocumentNormalizer:
            raise Exception("Document processors not available")

        logger.info(f"Starting processing for document {doc_id}: {original_filename}")

        # Check if JSON files already exist - if so, return existing data
        output_dir = Path(settings.output_dir)
        analysis_file = output_dir / f"{file_id}_analysis.json"
        
        if analysis_file.exists():
            logger.info(f"Analysis file already exists: {analysis_file}")
            with open(analysis_file, 'r', encoding='utf-8') as f:
                existing_result = json.load(f)
                return existing_result

        # 1) Normalize document using unified schema
        normalizer = DocumentNormalizer()
        normalized_obj = await asyncio.to_thread(normalizer.normalize_document, file_path)
        normalized = normalized_obj.to_dict()  # Convert dataclass to dict!
        
        logger.info(f"Document normalized: {normalized['summary']['total_pages']} pages, {normalized['summary']['total_elements']} elements")

        # 2) Extract raw data using specific readers
        suffix = Path(original_filename).suffix.lower()
        raw_extraction = {}
        
        if suffix == ".pptx" and PPTXReader:
            reader = PPTXReader()
            if reader.load_file(file_path):
                raw_extraction = reader.extract_to_dict()
                logger.info(f"PPTX processing: Found {len(raw_extraction.get('slides', []))} slides")
            else:
                logger.warning(f"PPTX failed to load: {file_path}")
                
        elif suffix == ".pdf" and PDFReader:
            reader = PDFReader()
            if reader.load_file(file_path):
                raw_extraction = reader.extract_to_dict()
                logger.info(f"PDF processing: Found {len(raw_extraction.get('pages', []))} pages")
            else:
                logger.warning(f"PDF failed to load: {file_path}")

        # 3) Build comprehensive processing result
        processing_result = {
            "success": True,
            "file_id": file_id,
            "doc_id": doc_id,
            "original_filename": original_filename,
            "user_info": user_info,
            "processed_at": normalized["normalized_at"],
            "document_analysis": normalized,
            "processing_summary": {
                "document_type": normalized["document_type"],
                "total_pages": normalized["summary"]["total_pages"],
                "total_elements": normalized["summary"]["total_elements"],
                "processing_time": "N/A"  # TODO: Add actual timing
            }
        }

        # 4) Save all JSON artifacts (ONLY IF THEY DON'T ALREADY EXIST)
        logger.info(f"Saving JSON files to: {output_dir}")
        
        # Get clean stem without extension
        clean_stem = Path(original_filename).stem
        clean_stem = "".join(c for c in clean_stem if c.isalnum() or c in " -_").strip().replace(" ", "_")

        # Create filenames with clean stem
        write_json(output_dir / f"{file_id}_{clean_stem}_analysis.json", processing_result)
        write_json(output_dir / f"{file_id}_{clean_stem}_normalized.json", normalized)
        write_json(output_dir / f"{file_id}_{clean_stem}_raw.json", raw_extraction)

        logger.debug(f"JSON files saved successfully:")
        logger.debug(f"  - {output_dir}/{file_id}_analysis.json")
        logger.debug(f"  - {output_dir}/{file_id}_normalized.json") 
        logger.debug(f"  - {output_dir}/{file_id}_raw.json")

        # 5) KEEP uploaded file - DO NOT DELETE IT
        logger.info(f"Uploaded file preserved at: {file_path}")

        return processing_result

    except Exception as e:
        logger.error(f"Processing error for document {doc_id}: {e}")
        
        return {
            "success": False,
            "error": str(e),
            "file_id": file_id,
            "doc_id": doc_id,
            "original_filename": original_filename
        }

@app.post("/test-llm")
async def test_llm():
    """Test LLM connection."""
    try:
        if settings.llm_provider == "groq":
            if not settings.groq_api_key:
                raise HTTPException(status_code=400, detail="Groq API key not configured")
            
            from groq import Groq
            client = Groq(api_key=settings.groq_api_key)
            
            response = await asyncio.to_thread(
                client.chat.completions.create,
                messages=[{"role": "user", "content": "Hello! Respond with just 'OK' if you can hear me."}],
                model=settings.groq_model,
                max_tokens=10
            )
            
            return {
                "status": "success",
                "provider": "groq",
                "model": settings.groq_model,
                "response": response.choices[0].message.content.strip()
            }
            
        else:
            return {"status": "error", "message": f"Provider {settings.llm_provider} not implemented yet"}
            
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/analysis-history")
async def get_analysis_history():
    """Get list of previous analysis results."""
    try:
        output_dir = Path(settings.output_dir)
        analysis_files = list(output_dir.glob("*_analysis.json"))
        
        history = []
        for file_path in sorted(analysis_files, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                summary = {
                    "file_id": data.get("file_id"),
                    "doc_id": data.get("doc_id"),
                    "original_filename": data.get("original_filename"),
                    "user_info": data.get("user_info"),
                    "processed_at": data.get("processed_at"),
                    "document_type": data.get("document_analysis", {}).get("document_type"),
                    "total_pages": data.get("processing_summary", {}).get("total_pages"),
                    "success": data.get("success", False)
                }
                history.append(summary)
            except Exception as e:
                logger.error(f"Error reading analysis file {file_path}: {e}")
                continue
        
        return {"history": history[:10]}  # Return last 10 analyses
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn_target = "backend.app:app" if settings.debug else app
    uvicorn.run(
        uvicorn_target,
        host="127.0.0.1",
        port=8000,
        reload=settings.debug
    )