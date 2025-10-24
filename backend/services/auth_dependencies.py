"""
FastAPI dependencies for authentication.
"""
import logging
from typing import Optional
from fastapi import Depends, HTTPException, status, Request, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ..database.database import get_db_sync
from ..database.models import User, UserRole
from .jwt_service import jwt_service

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


async def get_token_from_request(
    request: Request,
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session_cookie: Optional[str] = Cookie(None, alias="slide_review_session")
) -> Optional[str]:
    """
    Extract JWT token from request.
    Tries Authorization header first, then session cookie.

    Args:
        request: FastAPI request object
        authorization: Bearer token from Authorization header
        session_cookie: Session cookie value

    Returns:
        JWT token string or None
    """
    # Try Authorization header first
    if authorization:
        return authorization.credentials

    # Fall back to session cookie
    if session_cookie:
        return session_cookie

    return None


async def get_current_user(
    token: Optional[str] = Depends(get_token_from_request),
    db: Session = Depends(get_db_sync)
) -> User:
    """
    Get current authenticated user from JWT token.

    Args:
        token: JWT token
        db: Database session

    Returns:
        User object

    Raises:
        HTTPException: If authentication fails
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Decode token
    payload = jwt_service.decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify session exists and is valid
    session = jwt_service.get_session_by_token(db, token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.

    Args:
        current_user: User from get_current_user

    Returns:
        User object

    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify they have admin role.

    Args:
        current_user: User from get_current_user

    Returns:
        User object

    Raises:
        HTTPException: If user is not an admin
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def optional_user(
    token: Optional[str] = Depends(get_token_from_request),
    db: Session = Depends(get_db_sync)
) -> Optional[User]:
    """
    Get current user if authenticated, or None.
    Does not raise exception if not authenticated.

    Args:
        token: JWT token
        db: Database session

    Returns:
        User object or None
    """
    if not token:
        return None

    try:
        payload = jwt_service.decode_token(token)
        if not payload:
            return None

        user_id = payload.get("sub")
        user = db.query(User).filter(User.id == user_id).first()

        if user and user.is_active:
            return user

    except Exception as e:
        logger.warning(f"Error in optional_user: {e}")

    return None
