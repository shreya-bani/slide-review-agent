"""
FastAPI application entry point for Slide Review Agent.
"""
import asyncio
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from .config.settings import settings


try:
    from .processors.pptx_reader import PPTXReader
    from .processors.pdf_reader import PDFReader
    from .processors.document_normalizer import DocumentNormalizer
except Exception as e:

    print(f"Warning: Could not import document processors: {e}")
    PPTXReader = None
    PDFReader = None
    DocumentNormalizer = None


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

# Resolve frontend paths relative to the repository root (two levels up from
# this file: <repo>/backend/app.py -> repo root). This avoids issues when the
# current working directory differs (module vs script execution).
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
STYLES_DIR = FRONTEND_DIR / "styles"
SCRIPTS_DIR = FRONTEND_DIR / "scripts"
PAGES_DIR = FRONTEND_DIR / "pages"

# Mount static files for the frontend; raise a helpful error if missing so
# debugging is easier.
if not STYLES_DIR.exists() or not SCRIPTS_DIR.exists() or not FRONTEND_DIR.exists():
    print(f"Warning: frontend directories not found under {FRONTEND_DIR}. "
          "Static files endpoints may fail.")

app.mount("/styles", StaticFiles(directory=str(STYLES_DIR)), name="styles")
app.mount("/scripts", StaticFiles(directory=str(SCRIPTS_DIR)), name="scripts")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/app")
async def serve_frontend():
    """Serve the frontend application."""
    index_file = PAGES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail=f"Frontend index not found: {index_file}")
    return FileResponse(str(index_file))


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

@app.get("/debug-env")
async def debug_env():
    """Debug what environment values are being loaded."""
    return {
        "groq_api_key_exists": bool(settings.groq_api_key),
        "groq_api_key_length": len(settings.groq_api_key) if settings.groq_api_key else 0,
        "groq_api_key_first_4": settings.groq_api_key[:4] if settings.groq_api_key else None,
        "llm_provider": settings.llm_provider,
        "groq_model": settings.groq_model,
        "env_file_method": "load_dotenv",
        "current_working_dir": str(Path.cwd()),
        "processors_available": {
            "pptx_reader": PPTXReader is not None,
            "pdf_reader": PDFReader is not None,
            "document_normalizer": DocumentNormalizer is not None
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database path exists
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

from fastapi import Request

@app.post("/upload-document")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_info: Optional[str] = Form(None)
):
    user_info = user_info or "Anonymous"

    # 1) Validate extension
    allowed_extensions = ['.pptx', '.pdf']
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type. Allowed: {allowed_extensions}")

    # 2) Fast pre-check using Content-Length (if present)
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    cl = request.headers.get("content-length")
    if cl and int(cl) > max_bytes:
        raise HTTPException(400, f"File too large. Maximum size: {settings.max_file_size_mb}MB")

    # 3) Stream to disk and enforce limit while writing
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}_{file.filename}"
    file_path = Path(settings.upload_dir) / safe_filename

    written = 0
    chunk_size = 1024 * 1024  # 1MB
    with open(file_path, "wb") as buffer:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                buffer.close()
                try: os.remove(file_path)
                except: pass
                raise HTTPException(400, f"File too large. Maximum size: {settings.max_file_size_mb}MB")
            buffer.write(chunk)

    # 4) Process
    processing_result = await process_document_async(
        file_path=str(file_path),
        original_filename=file.filename,
        user_info=user_info,
        file_id=file_id
    )
    return JSONResponse(content=processing_result, status_code=200)



# async def process_document_async(file_path: str, original_filename: str, 
#                                user_info: str, file_id: str) -> dict:
#     """Process a document asynchronously."""
#     try:
#         if not DocumentNormalizer:
#             raise Exception("Document processors not available")
        
#         # Initialize normalizer
#         normalizer = DocumentNormalizer()
        
#         # Process document
#         result = await asyncio.to_thread(
#             normalizer.normalize_document, 
#             file_path
#         )
        
#         # Add processing metadata
#         processing_result = {
#             "success": True,
#             "file_id": file_id,
#             "original_filename": original_filename,
#             "user_info": user_info,
#             "processed_at": result["normalized_at"],
#             "document_analysis": result,
#             "processing_summary": {
#                 "document_type": result["document_type"],
#                 "total_pages": result["summary"]["total_pages"],
#                 "total_elements": result["summary"]["total_elements"],
#                 "processing_time": "N/A"  # TODO: Add actual timing
#             }
#         }
        
#         # Save result to output directory
#         output_file = Path(settings.output_dir) / f"{file_id}_analysis.json"
#         with open(output_file, 'w') as f:
#             import json
#             json.dump(processing_result, f, indent=2)
        
#         # Clean up uploaded file (optional)
#         try:
#             os.remove(file_path)
#         except:
#             pass
        
#         return processing_result
        
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#             "file_id": file_id,
#             "original_filename": original_filename
#         }

async def process_document_async(file_path: str, original_filename: str,
                                 user_info: str, file_id: str) -> dict:
    """Process a document asynchronously and save all JSON artifacts."""
    try:
        if not DocumentNormalizer:
            raise Exception("Document processors not available")

        # Small helper for tidy file writes
        def write_json(path: Path, obj: dict) -> None:
            import json
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)

        # 1) Normalize via the normalizer (your existing path)
        normalizer = DocumentNormalizer()
        normalized = await asyncio.to_thread(normalizer.normalize_document, file_path)

        # 2) Also capture RAW extraction directly from the specific reader
        suffix = Path(original_filename).suffix.lower()
        raw_extraction = {}
        if suffix == ".pptx" and PPTXReader:
            raw_extraction = await asyncio.to_thread(PPTXReader().extract, file_path)
        elif suffix == ".pdf" and PDFReader:
            raw_extraction = await asyncio.to_thread(PDFReader().extract, file_path)

        # 3) Build the combined response (normalized lives under document_analysis)
        processing_result = {
            "success": True,
            "file_id": file_id,
            "original_filename": original_filename,
            "user_info": user_info,
            "processed_at": normalized["normalized_at"],
            "document_analysis": normalized,
            "processing_summary": {
                "document_type": normalized["document_type"],
                "total_pages": normalized["summary"]["total_pages"],
                "total_elements": normalized["summary"]["total_elements"],
                "processing_time": "N/A"  # TODO: measure
            }
        }

        # 4) SAVE: combined, normalized-only, and raw
        output_dir = Path(settings.output_dir)
        write_json(output_dir / f"{file_id}_analysis.json", processing_result)
        write_json(output_dir / f"{file_id}_normalized.json", normalized)
        write_json(output_dir / f"{file_id}_raw.json", raw_extraction)

        # (Optional) Clean up upload to keep /uploads small
        try:
            os.remove(file_path)
        except Exception:
            pass

        return processing_result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_id": file_id,
            "original_filename": original_filename
        }


@app.post("/test-llm")
async def test_llm():
    """Test LLM connection."""
    try:
        if settings.llm_provider == "groq":
            if not settings.groq_api_key:
                raise HTTPException(status_code=400, detail="Groq API key not configured")
            
            # Import here to avoid startup issues if groq not installed
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
                with open(file_path, 'r') as f:
                    import json
                    data = json.load(f)
                    
                summary = {
                    "file_id": data.get("file_id"),
                    "original_filename": data.get("original_filename"),
                    "user_info": data.get("user_info"),
                    "processed_at": data.get("processed_at"),
                    "document_type": data.get("document_analysis", {}).get("document_type"),
                    "total_pages": data.get("processing_summary", {}).get("total_pages"),
                    "success": data.get("success", False)
                }
                history.append(summary)
            except:
                continue
        
        return {"history": history[:10]}  # Return last 10 analyses
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    if settings.debug:
        uvicorn_target = "backend.app:app"
    else:
        uvicorn_target = app

    uvicorn.run(
        uvicorn_target,
        host="127.0.0.1",
        port=8000,
        reload=settings.debug
    )