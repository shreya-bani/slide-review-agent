"""
FastAPI application with integrated document analysis pipeline.
"""
import asyncio
import os
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Response, Depends  # <-- added Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from starlette.responses import StreamingResponse
from logging import Handler, LogRecord

from .config.settings import settings
from .database.database import init_db, get_db_sync

# auth imports (new)
from .services.auth_dependencies import (
    get_current_user,
    get_current_active_user,
    get_current_admin_user,  # Admin-only routes protection
)
from .database.models import (
    User,
    Document,
    AnalysisResult,
    StyleIssue,
    AuditLog,
    DocumentType,
    IssueSeverity,
    IssueCategory
)
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import document processors
try:
    from .processors.pptx_reader import PPTXReader
    from .processors.pdf_reader import PDFReader
    from .processors.document_normalizer import DocumentNormalizer
except Exception as e:
    logger.warning(f"Could not import document processors: {e}")
    PPTXReader = None
    PDFReader = None
    DocumentNormalizer = None

# Import analyzers
try:
    from .analyzers.combine_analysis import CombinedAnalyzer
except Exception as e:
    logger.warning(f"Could not import analyzers: {e}")
    CombinedAnalyzer = None

# Logging handler to broadcast logs via asyncio queue
_broadcast_queue: "asyncio.Queue[dict]" = asyncio.Queue()
_clients: set[asyncio.Queue] = set()

class _QueueLogHandler(Handler):
    def emit(self, record: LogRecord) -> None:
        try:
            payload = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(timespec="seconds"),
                "level": record.levelname,
                "message": record.getMessage(),
                "component": record.name,
            }
            # Non-blocking put: if the loop isn't running yet, drop silently
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(_broadcast_queue.put_nowait, payload)
        except Exception:
            pass

q_handler = _QueueLogHandler()
q_handler.setLevel(logging.INFO)

# Optional: drop extremely chatty 'uvicorn.access' lines
class _NoAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("uvicorn.access")

q_handler.addFilter(_NoAccessFilter())

root_logger = logging.getLogger()           # root
if not any(isinstance(h, _QueueLogHandler) for h in root_logger.handlers):
    root_logger.addHandler(q_handler)
    root_logger.setLevel(logging.INFO)

# Global cache
processed_files = {}

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered slide deck review agent with style analysis",
        version="1.0.0",
        debug=settings.debug
    )
    
    # Parse CORS origins from settings (comma-separated or "*")
    cors_origins = ["*"] if settings.cors_origins == "*" else [
        origin.strip() for origin in settings.cors_origins.split(",")
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    settings.ensure_directories()
    return app

app = create_app()

# Initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize database and other startup tasks."""
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

# Include authentication router
try:
    from .routers.auth_router import router as auth_router
    app.include_router(auth_router)
    logger.info("Authentication router included")
except Exception as e:
    logger.warning(f"Could not include auth router: {e}")

# Include admin router
try:
    from .routers.admin_router import router as admin_router
    app.include_router(admin_router)
    logger.info("Admin router included")
except Exception as e:
    logger.warning(f"Could not include admin router: {e}")

# Frontend static files
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
STYLES_DIR = FRONTEND_DIR / "styles"
SCRIPTS_DIR = FRONTEND_DIR / "scripts"
PAGES_DIR = FRONTEND_DIR / "pages"

if STYLES_DIR.exists() and SCRIPTS_DIR.exists():
    app.mount("/styles", StaticFiles(directory=str(STYLES_DIR)), name="styles")
    app.mount("/scripts", StaticFiles(directory=str(SCRIPTS_DIR)), name="scripts")
    app.mount("/pages", StaticFiles(directory=str(PAGES_DIR)), name="pages")
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

def get_next_document_id() -> int:
    """Get the next incremental document ID."""
    output_dir = Path(settings.output_dir)
    if not output_dir.exists():
        return 1
    
    existing_ids = []
    for file_path in output_dir.glob("*_combined_analysis.json"):
        try:
            parts = file_path.stem.split('_')
            if parts and parts[0].isdigit():
                existing_ids.append(int(parts[0]))
        except:
            continue
    
    return max(existing_ids, default=0) + 1

def create_clean_filename(original_filename: str, doc_id: int) -> tuple[str, str]:
    """Create clean filename and file_id."""
    stem = Path(original_filename).stem
    extension = Path(original_filename).suffix.lower()
    
    clean_stem = "".join(c for c in stem if c.isalnum() or c in " -_").strip()
    clean_stem = clean_stem.replace(" ", "_")
    
    if len(clean_stem) > 50:
        clean_stem = clean_stem[:50]
    
    file_id = f"{doc_id:03d}"
    clean_filename = f"{file_id}_{clean_stem}{extension}"
    
    return clean_filename, file_id

def write_json(path: Path, obj: dict) -> None:
    """Write JSON files with proper formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# API Endpoints
@app.get("/")
async def root():
    """Redirect root to login page."""
    return RedirectResponse(url="/pages/login.html")

@app.get("/api")
async def api_root():
    """API endpoint - health check."""
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "environment": settings.environment
    }

# PROTECTED: user must be signed in
@app.get("/app")
async def serve_frontend(current_user: User = Depends(get_current_user)):
    """Serve the frontend application (protected)."""
    index_file = PAGES_DIR / "main.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index_file))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        db_path = Path(settings.database_path).parent
        db_accessible = db_path.exists() or db_path.parent.exists()
        
        dirs_exist = (
            Path(settings.upload_dir).exists() and 
            Path(settings.output_dir).exists()
        )
        
        llm_config_valid = settings.validate_llm_config()
        processors_available = all([PPTXReader, PDFReader, DocumentNormalizer])
        analyzers_available = CombinedAnalyzer is not None
        
        health_status = {
            "database": "ok" if db_accessible else "error",
            "directories": "ok" if dirs_exist else "error", 
            "llm_config": "ok" if llm_config_valid else "error",
            "processors": "ok" if processors_available else "error",
            "analyzers": "ok" if analyzers_available else "error",
            "overall": "healthy" if all([
                db_accessible, dirs_exist, llm_config_valid, 
                processors_available, analyzers_available
            ]) else "unhealthy"
        }
        
        status_code = 200 if health_status["overall"] == "healthy" else 503
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "overall": "unhealthy"}, 
            status_code=503
        )

# PROTECTED: signed-in & active user required
@app.post("/upload-document")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_info: str = Form(...),
    current_user: User = Depends(get_current_active_user),  # <-- added protection
    db: Session = Depends(get_db_sync),  # <-- database session
):
    """Upload and analyze a document with complete pipeline."""
    user_info = user_info.strip()
    if not user_info:
        raise HTTPException(400, "Error: Uploader name/email is required.")
    # Validate file
    allowed_extensions = ['.pptx', '.pdf']
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type. Allowed: {allowed_extensions}")

    # Check size
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    cl = request.headers.get("content-length")
    if cl and int(cl) > max_bytes:
        raise HTTPException(400, f"File too large. Maximum: {settings.max_file_size_mb}MB")

    # Generate filenames
    doc_id = get_next_document_id()
    clean_filename, file_id = create_clean_filename(file.filename, doc_id)
    file_path = Path(settings.upload_dir) / clean_filename

    logger.info(f"Processing document {doc_id}: {file.filename} -> {clean_filename}")

    # Save file
    written = 0
    chunk_size = 1024 * 1024
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
                    raise HTTPException(400, f"File too large. Maximum: {settings.max_file_size_mb}MB")
                buffer.write(chunk)
    except Exception as e:
        try:
            os.remove(file_path)
        except:
            pass
        raise HTTPException(500, f"Failed to save file: {str(e)}")

    # Create document record in database
    try:
        document_type = DocumentType.PPTX if file_extension == '.pptx' else DocumentType.PDF
        db_document = Document(
            file_id=file_id,
            original_filename=file.filename,
            clean_filename=clean_filename,
            document_type=document_type,
            file_size_bytes=written,
            user_id=current_user.id,
            user_info=user_info,
            upload_path=clean_filename,  # Relative path in upload dir
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        logger.info(f"Document record created in database: ID={db_document.id}, file_id={file_id}")

        # Log audit event
        audit_log = AuditLog(
            user_id=current_user.id,
            action="upload_document",
            resource_type="document",
            resource_id=file_id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"filename": file.filename, "size_bytes": written}
        )
        db.add(audit_log)
        db.commit()

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create document record: {e}")
        # Continue processing even if DB save fails
        db_document = None

    # Check cache
    if file_id in processed_files:
        logger.info(f"Document {file_id} already processed, returning cached result")
        cached_result = processed_files[file_id].copy()
        cached_result['user_info'] = user_info  # Update with current uploader
        cached_result['processed_at'] = datetime.now().isoformat()
        logger.info(f"Returning cached result with user_info: {cached_result.get('user_info')}")

        return JSONResponse(content=cached_result, status_code=200)

    # Process and analyze document
    processing_result = await process_and_analyze_document(
        file_path=str(file_path),
        original_filename=file.filename,
        user_info=user_info,
        file_id=file_id,
        doc_id=doc_id,
        db=db,
        db_document=db_document,
        current_user=current_user
    )

    # Cache result
    processed_files[file_id] = processing_result

    return JSONResponse(content=processing_result, status_code=200)

async def process_and_analyze_document(
    file_path: str,
    original_filename: str,
    user_info: str,
    file_id: str,
    doc_id: int,
    db: Session = None,
    db_document: Document = None,
    current_user: User = None
) -> dict:
    """Complete document processing and analysis pipeline."""
    try:
        if not DocumentNormalizer or not CombinedAnalyzer:
            raise Exception("Processors or analyzers not available")

        start_time = datetime.now()
        logger.info(f"Time processing started at {start_time.isoformat(timespec='seconds')}")

        logger.info(f"Starting complete analysis for document_id {file_id}")

        output_dir = Path(settings.output_dir)
        clean_stem = Path(original_filename).stem.replace(" ", "_")[:50]
        
        # Check if analysis already exists
        result_file = output_dir / f"{file_id}_{clean_stem}_result.json"
        if result_file.exists():
            logger.info(f"Result file already exists: {result_file}")
            with open(result_file, 'r', encoding='utf-8') as f:
                cached_result = json.load(f)
            cached_result['user_info'] = user_info  # Update with current uploader
            logger.info(f"Returning cached result with user_info: {cached_result.get('user_info')}")
            return cached_result

        # Step 1: Normalize document
        logger.info("Normalizing document into structured data format.")
        normalizer = DocumentNormalizer()
        normalized_obj = await asyncio.to_thread(normalizer.normalize_document, file_path)
        normalized_doc = normalized_obj.to_dict()
        
        logger.info(f"Normalized: {normalized_doc['summary']['total_pages']} pages, "
                   f"{normalized_doc['summary']['total_elements']} elements")

        # Step 2: Run combined analysis (grammar + tone + filename)
        logger.info("Running combined analysis according to amida guidelines.")
        analyzer = CombinedAnalyzer()
        analysis_report = await asyncio.to_thread(
            analyzer.analyze,
            normalized_doc,
            file_path,
            original_filename
        )
        
        logger.info(f"Analysis complete: {analysis_report['summary']['total_issues']} issues found")

        # Step 3: Build complete response
        processing_result = {
            "success": True,
            "file_id": file_id,
            "doc_id": doc_id,
            "original_filename": original_filename,
            "user_info": user_info,
            "processed_at": datetime.now().isoformat(),

            # Document metadata
            "metadata": analysis_report.get("document_metadata", {}),

            # Analysis summary
            "analysis_summary": {
                "total_issues": analysis_report["summary"]["total_issues"],
                "grammar_issues": analysis_report["summary"]["grammar_issues"],
                "tone_issues": analysis_report["summary"]["tone_issues"],
                "issues_with_suggestions": analysis_report["summary"]["issues_with_suggestions"],
                "severity_breakdown": analysis_report["summary"]["severity_breakdown"],
                "category_breakdown": analysis_report["summary"]["category_breakdown"],
                "rule_breakdown": analysis_report["summary"]["rule_breakdown"],
            },

            # Content statistics
            "content_statistics": analysis_report.get("content_statistics", {}),

            # All issues for table display
            "findings": analysis_report.get("all_issues", []),

            # Issues by slide for navigation
            "issues_by_slide": analysis_report.get("issues_by_slide", {}),

            # Categorized issues
            "issues_by_category": analysis_report.get("issues_by_category", {}),

            # Full analysis metadata
            "analysis_metadata": analysis_report.get("analysis_metadata", {}),
        }

        logger.info(f"Created processing_result with user_info: {processing_result.get('user_info')}")

        # Step 4: Save all outputs
        logger.info(f"Saving outputs to: {output_dir}")
        
        # Save combined analysis
        write_json(
            output_dir / f"{file_id}_{clean_stem}_combined_analysis.json",
            analysis_report
        )
        
        # Save normalized document
        write_json(
            output_dir / f"{file_id}_{clean_stem}_normalized.json",
            normalized_doc
        )
        
        # Save processing result (for quick frontend loading)
        write_json(
            output_dir / f"{file_id}_{clean_stem}_result.json",
            processing_result
        )

        # Timing log
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            f"Finished processing at {end_time.isoformat(timespec='seconds')} "
            f"(Duration: {duration:.2f}s)"
        )

        # Save analysis results to database
        if db and db_document:
            try:
                # Update document with processed timestamp and output path
                db_document.processed_at = end_time
                db_document.output_path = f"{file_id}_{clean_stem}_result.json"

                # Create analysis result record
                analysis_result = AnalysisResult(
                    document_id=db_document.id,
                    total_issues=analysis_report["summary"]["total_issues"],
                    grammar_issues=analysis_report["summary"]["grammar_issues"],
                    tone_issues=analysis_report["summary"]["tone_issues"],
                    issues_with_suggestions=analysis_report["summary"]["issues_with_suggestions"],
                    severity_breakdown=analysis_report["summary"]["severity_breakdown"],
                    category_breakdown=analysis_report["summary"]["category_breakdown"],
                    rule_breakdown=analysis_report["summary"]["rule_breakdown"],
                    total_pages=normalized_doc["summary"]["total_pages"],
                    total_elements=normalized_doc["summary"]["total_elements"],
                    processing_time_seconds=duration
                )
                db.add(analysis_result)
                db.commit()
                db.refresh(analysis_result)
                logger.info(f"Analysis result saved: ID={analysis_result.id}")

                # Create style issue records
                severity_map = {
                    # Database enum values
                    "critical": IssueSeverity.CRITICAL,
                    "high": IssueSeverity.HIGH,
                    "medium": IssueSeverity.MEDIUM,
                    "low": IssueSeverity.LOW,
                    "info": IssueSeverity.INFO,
                    # Analyzer enum values -> map to database equivalents
                    "error": IssueSeverity.HIGH,
                    "warning": IssueSeverity.MEDIUM,
                    "suggestion": IssueSeverity.LOW
                }
                category_map = {
                    "grammar": IssueCategory.GRAMMAR,
                    "tone": IssueCategory.TONE,
                    "tone-issue": IssueCategory.TONE,  # Handle "tone-issue" from analyzer
                    "style": IssueCategory.STYLE,
                    "formatting": IssueCategory.FORMATTING,
                    "content": IssueCategory.CONTENT,
                    "filename": IssueCategory.STYLE  # Map filename issues to STYLE category
                }

                for issue in analysis_report.get("all_issues", []):
                    try:
                        style_issue = StyleIssue(
                            analysis_id=analysis_result.id,
                            category=category_map.get(issue.get("category", "grammar"), IssueCategory.GRAMMAR),
                            severity=severity_map.get(issue.get("severity", "medium"), IssueSeverity.MEDIUM),
                            rule_name=issue.get("rule_name", "unknown"),
                            slide_number=issue.get("page_or_slide_index"),
                            element_type=issue.get("element_type"),
                            element_index=issue.get("element_index"),
                            original_text=issue.get("found_text", ""),
                            suggested_text=issue.get("suggestion"),
                            explanation=issue.get("description")
                        )
                        db.add(style_issue)
                    except Exception as e:
                        logger.error(f"Failed to create style issue record: {e}")
                        continue

                db.commit()
                logger.info(f"Saved {len(analysis_report.get('all_issues', []))} style issues to database")

                # Log audit event for analysis completion
                if current_user:
                    audit_log = AuditLog(
                        user_id=current_user.id,
                        action="analyze_document",
                        resource_type="document",
                        resource_id=file_id,
                        details={
                            "total_issues": analysis_report["summary"]["total_issues"],
                            "processing_time": duration
                        }
                    )
                    db.add(audit_log)
                    db.commit()

            except Exception as e:
                db.rollback()
                logger.error(f"Failed to save analysis results to database: {e}")
                import traceback
                traceback.print_exc()

        logger.info(f"Processing complete for document_id {file_id}")
        return processing_result

    except Exception as e:
        logger.error(f"Processing error for document_id {file_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "file_id": file_id,
            "doc_id": doc_id,
            "original_filename": original_filename
        }

# PROTECTED: signed-in user required
@app.get("/analysis-history")
async def get_analysis_history(current_user: User = Depends(get_current_user)):
    """Get list of previous analyses (protected)."""
    try:
        output_dir = Path(settings.output_dir)
        analysis_files = list(output_dir.glob("*_combined_analysis.json"))
        
        history = []
        for file_path in sorted(analysis_files, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                summary = {
                    "file_id": file_path.stem.split('_')[0],
                    "filename": file_path.name,
                    "processed_at": data.get("analysis_metadata", {}).get("timestamp"),
                    "total_issues": data.get("summary", {}).get("total_issues", 0),
                    "total_pages": data.get("document_metadata", {}).get("total_pages"),
                }
                history.append(summary)
            except Exception as e:
                logger.error(f"Error reading analysis file {file_path}: {e}")
                continue
        
        return {"history": history[:10]}
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

async def _fanout_logs():
    while True:
        item = await _broadcast_queue.get()
        for q in list(_clients):
            # each client has its own queue; don't await here
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                pass

# Start the fanout task on startup
@app.on_event("startup")
async def _startup_bg():
    asyncio.create_task(_fanout_logs())

# PROTECTED: Admin-only endpoint to stream logs
@app.get("/logs/stream")
async def logs_stream(current_admin: User = Depends(get_current_admin_user)):
    """Stream server logs in real-time. Admin only."""
    client_q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _clients.add(client_q)

    async def event_gen():
        try:
            yield "event: hello\ndata: {}\n\n"
            while True:
                item = await client_q.get()
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            _clients.discard(client_q)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"  # Nginx: disable buffering
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)
