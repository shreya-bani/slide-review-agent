"""
Authentication router - handles Azure AD login/logout flows.
"""
import logging
import secrets
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from ..database.database import get_db
from ..config.azure_config import azure_settings
from ..services.auth_service import azure_ad_service
from ..services.jwt_service import jwt_service
from ..services.auth_dependencies import get_current_user, optional_user
from ..database.models import User, UserRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.get("/login")
async def login(request: Request):
    """
    Initiate Azure AD login flow.
    Returns the Azure AD authorization URL for the frontend to redirect to.
    """
    try:
        logger.info("----------------------------LOGIN INITIATED...")

        # Validate Azure configuration
        if not azure_settings.validate_azure_config():
            logger.error("Azure AD configuration validation failed!")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Azure AD is not properly configured!"
            )

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        logger.info(f"Generated state: {state}")

        # Get authorization URL with PKCE
        auth_url, flow = azure_ad_service.get_auth_url_with_pkce(state=state)
        logger.info(f"Generated auth URL with PKCE: {auth_url}")
        logger.info(f"Redirect URI configured: {azure_settings.azure_redirect_uri}")

        return JSONResponse(content={
            "auth_url": auth_url,
            "state": state
        })

    except Exception as e:
        logger.error(f"Login initiation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate login"
        )


@router.get("/callback")
async def auth_callback(
    request: Request,
    response: Response,
    code: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    state: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Azure AD OAuth callback endpoint.
    Handles the redirect from Azure AD after user authentication.
    """
    logger.info("-------------------------CALLBACK RECEIVED!")
    logger.info(f"Code present: {code is not None}")
    logger.info(f"Error: {error}")
    logger.info(f"Query params: {dict(request.query_params)}")
    logger.info(f"Full URL: {request.url}")

    # 0) Handle Azure errors early
    if error:
        logger.error(f"AZURE AD ERROR: {error} - {error_description} ===")
        return RedirectResponse(
            url=f"/pages/login.html?error={error}&error_description={error_description}",
            status_code=status.HTTP_302_FOUND
        )

    # 1) Require code + state
    if not code:
        logger.error("No authorization code received")
        return RedirectResponse(
            url="/pages/login.html?error=no_code",
            status_code=status.HTTP_302_FOUND
        )
    if not state:
        logger.error("Missing state parameter")
        return RedirectResponse(
            url="/pages/login.html?error=missing_state",
            status_code=status.HTTP_302_FOUND
        )

    try:
        logger.info("------------------ EXCHANGING CODE FOR TOKEN....")

        # 2) Exchange code with PKCE & verify state (azure_ad_service should validate state internally)
        token_response = azure_ad_service.acquire_token_by_auth_code(code, state)
        if not token_response:
            logger.error("Token response is None!")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to acquire access token"
            )
        logger.info(f"Token response keys: {list(token_response.keys())}")

        # 3) Claims & token
        id_token_claims = token_response.get("id_token_claims", {}) or {}
        access_token = token_response.get("access_token")

        # 4) Resolve groups: prefer ID token claim, fallback to Graph
        groups = azure_ad_service.get_user_groups_from_token(id_token_claims)
        if not groups and access_token:
            groups = await azure_ad_service.get_user_groups(access_token)

        # 5) Enforce membership in allowed groups (hard gate)
        if not azure_ad_service.validate_user_access(groups):
            logger.warning("Access denied: user not in allowed groups!")
            return RedirectResponse(
                url="/pages/login.html?error=access_denied",
                status_code=status.HTTP_302_FOUND
            )

        # 6) Map groups → role (ADMIN or USER)
        user_role = azure_ad_service.determine_user_role(groups)
        logger.info(f"Assigned role from groups: {user_role}")

        # 7) Extract user info
        azure_oid = id_token_claims.get("oid")
        email = id_token_claims.get("email") or id_token_claims.get("preferred_username")
        display_name = id_token_claims.get("name", "Unknown User")
        logger.info(f"Extracted - OID: {azure_oid}, Email: {email}, Name: {display_name}")

        if not azure_oid or not email:
            logger.error("Missing required user information")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user information from Azure AD"
            )

        # 8) Create or update user with the resolved role  ❗️(do NOT overwrite role below)
        user = azure_ad_service.get_or_create_user(
            db=db,
            azure_oid=azure_oid,
            email=email,
            display_name=display_name,
            role=user_role,  # <-- keep role from groups
        )

        # 9) JWT + session
        logger.info("----------------------- GENERATING JWT TOKEN....")
        jwt_token = jwt_service.create_access_token(user)
        # Avoid logging token contents to reduce leakage
        logger.info("JWT token generated")

        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        logger.info(f"Creating session for user {email} from IP {client_ip}")

        session = jwt_service.create_session(
            db=db,
            user=user,
            token=jwt_token,
            ip_address=client_ip,
            user_agent=user_agent
        )
        logger.info(f"Session created: {session.id}")

        # 10) Set cookie & redirect to the protected app shell
        response = RedirectResponse(url="/app", status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key=azure_settings.session_cookie_name,
            value=jwt_token,
            max_age=azure_settings.session_expiry_hours * 3600,
            httponly=True,
            secure=azure_settings.session_cookie_secure,
            samesite="lax"
        )

        logger.info(f"----------------- AUTHENTICATION COMPLETE FOR {email}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication callback error: {e}", exc_info=True)
        return RedirectResponse(
            url="/pages/login.html?error=auth_failed&error_description=Authentication failed. Please try again.",
            status_code=status.HTTP_302_FOUND
        )



@router.get("/me")
async def get_current_user_info(
    current_user: Optional[User] = Depends(optional_user)
):
    """
    Get current authenticated user information.
    Returns user data if authenticated, or indication if not.
    """
    if not current_user:
        return JSONResponse(
            content={"authenticated": False},
            status_code=status.HTTP_200_OK
        )

    return JSONResponse(content={
        "authenticated": True,
        "user": current_user.to_dict()
    })


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Logout current user - invalidate session and clear cookie.
    """
    try:
        # Get token from cookie or header
        token = request.cookies.get(azure_settings.session_cookie_name)
        if not token:
            # Try Authorization header
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer"):
                token = auth_header[7:]

        # Delete session from database
        if token:
            jwt_service.delete_session(db, token)

        # Clear cookie
        response = JSONResponse(content={"message": "Logged out successfully"})
        response.delete_cookie(key=azure_settings.session_cookie_name)

        logger.info(f"User {current_user.email} logged out")
        return response

    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/health")
async def auth_health():
    """
    Check authentication service health.
    """
    azure_configured = azure_settings.validate_azure_config()

    return JSONResponse(content={
        "azure_ad_configured": azure_configured,
        "tenant_id": azure_settings.azure_tenant_id if azure_configured else None,
        "redirect_uri": azure_settings.azure_redirect_uri
    })
