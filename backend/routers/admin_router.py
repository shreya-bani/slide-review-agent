"""
Admin-only API endpoints for analytics, user management, and system monitoring.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from ..database.database import get_db_sync
from ..database.models import User, Session as DBSession, UserRole, Document, AnalysisResult
from ..services.auth_dependencies import get_current_admin_user
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/analytics/overview")
async def get_analytics_overview(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db_sync)
) -> Dict[str, Any]:
    """
    Get high-level analytics overview.
    Admin only.
    """
    try:
        # Total users
        total_users = db.query(func.count(User.id)).scalar()
        active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()

        # Users by role
        admin_count = db.query(func.count(User.id)).filter(User.role == UserRole.ADMIN).scalar()
        user_count = db.query(func.count(User.id)).filter(User.role == UserRole.USER).scalar()

        # Active sessions
        now = datetime.utcnow()
        active_sessions = db.query(func.count(DBSession.id)).filter(
            DBSession.expires_at > now
        ).scalar()

        # Users logged in last 24 hours
        last_24h = now - timedelta(hours=24)
        recent_logins = db.query(func.count(User.id)).filter(
            User.last_login >= last_24h
        ).scalar()

        # Users logged in last 7 days
        last_7d = now - timedelta(days=7)
        weekly_logins = db.query(func.count(User.id)).filter(
            User.last_login >= last_7d
        ).scalar()

        # Get document counts from database
        total_documents = db.query(func.count(Document.id)).scalar()
        processed_documents = db.query(func.count(Document.id)).filter(
            Document.processed_at != None
        ).scalar()

        # Analysis statistics
        total_analyses = db.query(func.count(AnalysisResult.id)).scalar()
        total_issues = db.query(func.sum(AnalysisResult.total_issues)).scalar()

        return {
            "users": {
                "total": total_users or 0,
                "active": active_users or 0,
                "inactive": (total_users or 0) - (active_users or 0),
                "admins": admin_count or 0,
                "regular_users": user_count or 0
            },
            "sessions": {
                "active_now": active_sessions or 0
            },
            "activity": {
                "logins_last_24h": recent_logins or 0,
                "logins_last_7d": weekly_logins or 0
            },
            "documents": {
                "total_uploaded": total_documents or 0,
                "processed": processed_documents or 0,
                "pending": (total_documents or 0) - (processed_documents or 0)
            },
            "analysis": {
                "total_analyses": total_analyses or 0,
                "total_issues_found": int(total_issues) if total_issues else 0
            },
            "system": {
                "uptime_hours": "N/A",  # Can be implemented with process tracking
                "environment": settings.environment
            }
        }
    except Exception as e:
        logger.error(f"Error fetching analytics overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch analytics"
        )


@router.get("/users")
async def list_all_users(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db_sync),
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    List all users with pagination.
    Admin only.
    """
    try:
        # Get total count
        total = db.query(func.count(User.id)).scalar()

        # Get users with pagination
        users = db.query(User).order_by(desc(User.last_login)).limit(limit).offset(offset).all()

        return {
            "total": total or 0,
            "limit": limit,
            "offset": offset,
            "users": [user.to_dict() for user in users]
        }
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch users"
        )


@router.get("/users/recent-logins")
async def get_recent_logins(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db_sync),
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Get recent user logins.
    Admin only.
    """
    try:
        users = db.query(User).order_by(desc(User.last_login)).limit(limit).all()

        return [{
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "role": user.role.value,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "is_active": user.is_active
        } for user in users]
    except Exception as e:
        logger.error(f"Error fetching recent logins: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch recent logins"
        )


@router.get("/sessions/active")
async def get_active_sessions(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db_sync),
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get all active sessions.
    Admin only.
    """
    try:
        now = datetime.utcnow()
        sessions = db.query(DBSession).filter(
            DBSession.expires_at > now
        ).order_by(desc(DBSession.created_at)).limit(limit).all()

        result = []
        for session in sessions:
            user = db.query(User).filter(User.id == session.user_id).first()
            result.append({
                "session_id": session.id,
                "user_email": user.email if user else "Unknown",
                "user_name": user.display_name if user else "Unknown",
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                "ip_address": session.ip_address,
                "user_agent": session.user_agent[:100] if session.user_agent else None  # Truncate
            })

        return result
    except Exception as e:
        logger.error(f"Error fetching active sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch active sessions"
        )


@router.get("/documents/recent")
async def get_recent_documents(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db_sync),
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get recent document uploads from database.
    Admin only.
    """
    try:
        # Query documents from database, ordered by upload time
        documents = db.query(Document).order_by(desc(Document.uploaded_at)).limit(limit).all()

        result = []
        for doc in documents:
            # Get associated user if available
            user = db.query(User).filter(User.id == doc.user_id).first() if doc.user_id else None

            # Get analysis results if available
            analysis = db.query(AnalysisResult).filter(
                AnalysisResult.document_id == doc.id
            ).first()

            result.append({
                "id": doc.id,
                "file_id": doc.file_id,
                "filename": doc.original_filename,
                "clean_filename": doc.clean_filename,
                "type": doc.document_type.value,
                "size_bytes": doc.file_size_bytes,
                "size_mb": round(doc.file_size_bytes / (1024 * 1024), 2) if doc.file_size_bytes else 0,
                "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                "is_processed": doc.processed_at is not None,
                "uploaded_by": {
                    "email": user.email if user else None,
                    "display_name": user.display_name if user else None,
                    "user_info": doc.user_info  # Legacy uploader info
                },
                "analysis": {
                    "total_issues": analysis.total_issues if analysis else None,
                    "grammar_issues": analysis.grammar_issues if analysis else None,
                    "tone_issues": analysis.tone_issues if analysis else None,
                    "processing_time": analysis.processing_time_seconds if analysis else None
                } if analysis else None
            })

        return result
    except Exception as e:
        logger.error(f"Error fetching recent documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch recent documents"
        )


@router.post("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: str,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db_sync)
) -> Dict[str, Any]:
    """
    Deactivate a user account.
    Admin only.
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Prevent admin from deactivating themselves
        if user.id == current_admin.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )

        user.is_active = False
        db.commit()

        logger.info(f"Admin {current_admin.email} deactivated user {user.email}")

        return {
            "success": True,
            "message": f"User {user.email} deactivated"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating user: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate user"
        )


@router.post("/users/{user_id}/activate")
async def activate_user(
    user_id: str,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db_sync)
) -> Dict[str, Any]:
    """
    Activate a user account.
    Admin only.
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user.is_active = True
        db.commit()

        logger.info(f"Admin {current_admin.email} activated user {user.email}")

        return {
            "success": True,
            "message": f"User {user.email} activated"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating user: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate user"
        )
