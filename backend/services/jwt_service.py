"""
JWT token generation and validation service.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from sqlalchemy.orm import Session

from ..config.azure_config import azure_settings
from ..database.models import User, Session as DBSession

logger = logging.getLogger(__name__)


class JWTService:
    """Service for handling JWT token operations."""

    def __init__(self):
        """Initialize JWT service with configuration."""
        self.secret_key = azure_settings.jwt_secret_key
        self.algorithm = azure_settings.jwt_algorithm
        self.expiry_minutes = azure_settings.jwt_expiry_minutes

    def create_access_token(self, user: User) -> str:
        """
        Create a JWT access token for the user.

        Args:
            user: User object

        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=self.expiry_minutes)

        payload = {
            "sub": user.id,  # Subject (user ID)
            "email": user.email,
            "display_name": user.display_name,
            "role": user.role.value,
            "azure_oid": user.azure_oid,
            "iat": now,  # Issued at
            "exp": expires_at,  # Expiration time
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created JWT token for user: {user.email}")
            return token
        except Exception as e:
            logger.error(f"Error creating JWT token: {e}")
            raise

    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload dict or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error decoding token: {e}")
            return None

    def verify_token(self, token: str) -> bool:
        """
        Verify if a token is valid.

        Args:
            token: JWT token string

        Returns:
            True if token is valid and not expired
        """
        return self.decode_token(token) is not None

    def create_session(
        self,
        db: Session,
        user: User,
        token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> DBSession:
        """
        Create a new session record in the database.

        Args:
            db: Database session
            user: User object
            token: JWT token
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Session object
        """
        expires_at = datetime.utcnow() + timedelta(hours=azure_settings.session_expiry_hours)

        session = DBSession(
            user_id=user.id,
            token=token,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        logger.info(f"Created session for user: {user.email}")
        return session

    def get_session_by_token(self, db: Session, token: str) -> Optional[DBSession]:
        """
        Get session by token.

        Args:
            db: Database session
            token: JWT token

        Returns:
            Session object or None
        """
        session = db.query(DBSession).filter(DBSession.token == token).first()

        if session and session.is_expired():
            logger.info(f"Session expired, deleting: {session.id}")
            db.delete(session)
            db.commit()
            return None

        return session

    def delete_session(self, db: Session, token: str) -> bool:
        """
        Delete a session (logout).

        Args:
            db: Database session
            token: JWT token

        Returns:
            True if session was deleted
        """
        session = db.query(DBSession).filter(DBSession.token == token).first()

        if session:
            db.delete(session)
            db.commit()
            logger.info(f"Deleted session: {session.id}")
            return True

        return False

    def cleanup_expired_sessions(self, db: Session) -> int:
        """
        Clean up all expired sessions.

        Args:
            db: Database session

        Returns:
            Number of sessions deleted
        """
        now = datetime.utcnow()
        expired_sessions = db.query(DBSession).filter(DBSession.expires_at < now).all()

        count = len(expired_sessions)
        for session in expired_sessions:
            db.delete(session)

        db.commit()
        logger.info(f"Cleaned up {count} expired sessions")
        return count


# Create global instance
jwt_service = JWTService()
