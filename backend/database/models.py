"""
Database models for authentication and user management.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum
import uuid

from .database import Base


class UserRole(str, enum.Enum):
    """User roles enum."""
    USER = "USER"
    ADMIN = "ADMIN"


class User(Base):
    """
    User model - stores authenticated users from Azure AD.
    """
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    azure_oid = Column(String(255), unique=True, nullable=False, index=True)  # Azure Object ID
    email = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.USER)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"

    def to_dict(self):
        """Convert user to dictionary."""
        return {
            "id": self.id,
            "azure_oid": self.azure_oid,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active
        }


class Session(Base):
    """
    Session model - stores active user sessions.
    """
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token = Column(String(512), unique=True, nullable=False, index=True)  # JWT token
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ip_address = Column(String(45), nullable=True)  # IPv6 max length
    user_agent = Column(String(512), nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

    def __repr__(self):
        return f"<Session(id={self.id}, user_id={self.user_id}, expires_at={self.expires_at})>"

    def is_expired(self):
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at

    def to_dict(self):
        """Convert session to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ip_address": self.ip_address,
            "is_expired": self.is_expired()
        }
