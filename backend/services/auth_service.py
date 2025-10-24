"""
Azure AD authentication service using MSAL.
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import msal
from sqlalchemy.orm import Session

from ..config.azure_config import azure_settings
from ..database.models import User, UserRole, Session as DBSession
from ..database.database import get_db

logger = logging.getLogger(__name__)

# Global storage for PKCE verifiers (in production, use Redis or database)
_pkce_verifiers = {}


class AzureADService:
    """
    Service for handling Azure AD authentication using MSAL (Microsoft Authentication Library).
    """

    def __init__(self):
        """Initialize the Azure AD service."""
        self.tenant_id = azure_settings.azure_tenant_id
        self.client_id = azure_settings.azure_client_id
        self.client_secret = azure_settings.azure_client_secret
        self.redirect_uri = azure_settings.azure_redirect_uri
        self.authority = azure_settings.authority
        self.group_role_mapping = azure_settings.get_group_role_mapping()

        # Create MSAL confidential client app (for Web application with client secret)
        # Web apps in Azure AD are confidential clients and require client_secret
        self.app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=self.authority
        )

    def get_auth_url_with_pkce(self, state: Optional[str] = None) -> tuple[str, dict]:
        """
        Get the Azure AD authorization URL for user login with PKCE support.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Tuple of (authorization URL, flow data containing code_verifier)
        """
        try:
            # Use empty scopes - we only need OpenID Connect (automatic)
            scopes = []

            # Use MSAL's built-in PKCE support via initiate_auth_code_flow
            flow = self.app.initiate_auth_code_flow(
                scopes=scopes,
                redirect_uri=self.redirect_uri,
                state=state
            )

            if "error" in flow:
                logger.error(f"Error initiating auth flow: {flow.get('error_description', flow['error'])}")
                raise Exception(f"Failed to initiate auth flow: {flow['error']}")

            # Store the entire flow (contains code_verifier) keyed by state
            _pkce_verifiers[state] = flow
            logger.info(f"Generated PKCE flow for state: {state}")
            logger.info(f"Code verifier present: {'code_verifier' in flow}")

            auth_url = flow["auth_uri"]
            logger.info(f"Generated Azure AD auth URL with PKCE")
            logger.info(f"Full auth URL: {auth_url}")

            return auth_url, flow
        except Exception as e:
            logger.error(f"Error generating auth URL: {e}")
            raise

    def acquire_token_by_auth_code(self, auth_code: str, state: str) -> Optional[Dict[str, Any]]:
        """
        Exchange authorization code for access token with PKCE.

        Args:
            auth_code: Authorization code received from Azure AD callback
            state: State parameter to retrieve PKCE flow

        Returns:
            Token response containing access_token, id_token, etc.
        """
        try:
            # Retrieve the stored flow (contains code_verifier and other PKCE data)
            flow = _pkce_verifiers.pop(state, None)
            if not flow:
                logger.error(f"No PKCE flow found for state: {state}")
                return None

            logger.info(f"Using PKCE flow for state: {state}")
            logger.info(f"Flow keys: {list(flow.keys())}")

            # Use MSAL's acquire_token_by_auth_code_flow which handles PKCE automatically
            result = self.app.acquire_token_by_auth_code_flow(
                auth_code_flow=flow,
                auth_response={"code": auth_code, "state": state}
            )

            if "error" in result:
                logger.error(f"Token acquisition error: {result.get('error_description', result['error'])}")
                return None

            logger.info("Successfully acquired access token")
            return result

        except Exception as e:
            logger.error(f"Exception during token acquisition: {e}")
            return None

    def get_user_groups_from_token(self, id_token_claims: dict) -> list[str]:
        """
        Get user's group memberships from ID token claims.

        Note: This requires the Azure AD app to include group claims in the token.
        Configure this in Azure Portal: Token Configuration -> Add groups claim

        Args:
            id_token_claims: ID token claims from Azure AD

        Returns:
            List of group Object IDs the user belongs to
        """
        # Try to get groups from token claims
        groups = id_token_claims.get("groups", [])

        if groups:
            logger.info(f"Found {len(groups)} group memberships in token")
            return groups

        # If no groups in token, log warning
        logger.warning("No groups found in ID token. You may need to configure 'groups' claim in Azure AD app registration.")
        return []

    async def get_user_groups(self, access_token: str) -> list[str]:
        """
        Get user's group memberships from Microsoft Graph API.
        (Fallback method - requires GroupMember.Read.All permission)

        Args:
            access_token: Access token with GroupMember.Read.All scope

        Returns:
            List of group Object IDs the user belongs to
        """
        import httpx

        headers = {"Authorization": f"Bearer {access_token}"}
        graph_url = "https://graph.microsoft.com/v1.0/me/memberOf"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(graph_url, headers=headers)
                response.raise_for_status()
                data = response.json()

                # Extract group Object IDs
                group_ids = [
                    item["id"]
                    for item in data.get("value", [])
                    if item.get("@odata.type") == "#microsoft.graph.group"
                ]

                logger.info(f"Found {len(group_ids)} group memberships")
                return group_ids

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch user groups: {e}")
            return []
        except Exception as e:
            logger.error(f"Exception fetching user groups: {e}")
            return []

    def determine_user_role(self, group_ids: list[str]) -> UserRole:
        """
        Determine user role based on Azure AD group membership.

        Args:
            group_ids: List of group Object IDs user belongs to

        Returns:
            UserRole (ADMIN or USER)
        """
        # Check if user is in admin group
        if azure_settings.azure_group_admins in group_ids:
            return UserRole.ADMIN

        # Check if user is in user group
        if azure_settings.azure_group_users in group_ids:
            return UserRole.USER

        # Default to USER if in any configured group
        logger.warning(f"User not in expected groups, defaulting to USER role")
        return UserRole.USER

    def validate_user_access(self, group_ids: list[str]) -> bool:
        """
        Validate if user has access to the application.

        Args:
            group_ids: List of group Object IDs user belongs to

        Returns:
            True if user is member of any configured group
        """
        required_groups = [
            azure_settings.azure_group_admins,
            azure_settings.azure_group_users
        ]

        has_access = any(group_id in group_ids for group_id in required_groups)

        # TEMPORARY: Allow access if no groups available (for testing)
        # Remove this in production after admin consent is granted
        if not group_ids:
            logger.warning("No group information available - granting temporary access for testing")
            return True

        if not has_access:
            logger.warning(f"User not in any required groups")

        return has_access

    def get_or_create_user(
        self,
        db: Session,
        azure_oid: str,
        email: str,
        display_name: str,
        role: UserRole
    ) -> User:
        """
        Get existing user or create new one.

        Args:
            db: Database session
            azure_oid: Azure Object ID
            email: User email
            display_name: User display name
            role: User role

        Returns:
            User object
        """
        # Try to find existing user
        user = db.query(User).filter(User.azure_oid == azure_oid).first()

        if user:
            # Update existing user
            user.email = email
            user.display_name = display_name
            user.role = role
            user.last_login = datetime.utcnow()
            user.is_active = True
            logger.info(f"Updated existing user: {email}")
        else:
            # Create new user
            user = User(
                azure_oid=azure_oid,
                email=email,
                display_name=display_name,
                role=role,
                is_active=True
            )
            db.add(user)
            logger.info(f"Created new user: {email}")

        db.commit()
        db.refresh(user)
        return user


# Create global instance
azure_ad_service = AzureADService()
