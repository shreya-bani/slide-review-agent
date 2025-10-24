"""
Azure Entra ID (Azure AD) configuration.
"""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class AzureADSettings(BaseSettings):
    """
    Azure AD authentication settings.

    All values should be set in .env.example file.
    """

    # Azure AD App Registration
    azure_tenant_id: str = Field(default="", alias="AZURE_TENANT_ID")
    azure_client_id: str = Field(default="", alias="AZURE_CLIENT_ID")
    azure_client_secret: str = Field(default="", alias="AZURE_CLIENT_SECRET")
    azure_redirect_uri: str = Field(default="http://127.0.0.1:8000/auth/callback", alias="AZURE_REDIRECT_URI")

    # Azure AD Groups (Object IDs)
    azure_group_users: str = Field(default="", alias="AZURE_GROUP_USERS")
    azure_group_admins: str = Field(default="", alias="AZURE_GROUP_ADMINS")

    # JWT Configuration
    jwt_secret_key: str = Field(default="", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expiry_minutes: int = Field(default=1440, alias="JWT_EXPIRY_MINUTES")  # 24 hours

    # Session Configuration
    session_expiry_hours: int = Field(default=24, alias="SESSION_EXPIRY_HOURS")
    session_cookie_name: str = Field(default="slide_review_session", alias="SESSION_COOKIE_NAME")
    session_cookie_secure: bool = Field(default=False, alias="SESSION_COOKIE_SECURE")  # Set to True in production with HTTPS

    class Config:
        env_file = ".env.example"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env file

    def validate_azure_config(self) -> bool:
        """
        Validate that all required Azure AD settings are configured.

        Note: client_secret IS required for Web application (confidential client).

        Returns:
            True if all required settings are present, False otherwise
        """
        required_fields = [
            self.azure_tenant_id,
            self.azure_client_id,
            self.azure_client_secret,  # Required for Web application
            self.azure_group_users,
            self.azure_group_admins,
        ]
        return all(field and field.strip() for field in required_fields)

    @property
    def authority(self) -> str:
        """Get the Azure AD authority URL."""
        return f"https://login.microsoftonline.com/{self.azure_tenant_id}"

    @property
    def scope(self) -> list[str]:
        """Get the OAuth scopes to request."""
        return [
            "openid",              # OpenID Connect
            "profile",             # User profile
            "email",               # User email
            "User.Read"            # Read user profile (doesn't require admin consent)
        ]

    def get_group_role_mapping(self) -> dict[str, str]:
        """
        Get mapping of Azure AD group IDs to application roles.

        Returns:
            Dictionary mapping group Object IDs to role names
        """
        return {
            self.azure_group_admins: "ADMIN",
            self.azure_group_users: "USER",
        }


# Create global instance
azure_settings = AzureADSettings()
