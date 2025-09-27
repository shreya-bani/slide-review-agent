"""
FastAPI application entry point for Slide Review Agent.
"""
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from config.settings import settings


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

# Mount static files for frontend
# Mount static files for frontend
app.mount("/styles", StaticFiles(directory="../frontend/styles"), name="styles")
app.mount("/scripts", StaticFiles(directory="../frontend/scripts"), name="scripts")
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/app")
async def serve_frontend():
    """Serve the frontend application."""
    return FileResponse("../frontend/pages/index.html")


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
        "env_file_method": "load_dotenv",  # Changed this line
        "current_working_dir": str(Path.cwd())
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
        
        health_status = {
            "database": "ok" if db_accessible else "error",
            "directories": "ok" if dirs_exist else "error", 
            "llm_config": "ok" if llm_config_valid else "error",
            "overall": "healthy" if all([db_accessible, dirs_exist, llm_config_valid]) else "unhealthy"
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
        "has_openai_key": bool(settings.openai_api_key)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.debug
    )