/**
 * Authentication handling for Azure AD SSO
 */

const API_BASE_URL = window.location.origin;

// DOM Elements
const microsoftLoginBtn = document.getElementById('microsoft-login-btn');
const loginStatus = document.getElementById('login-status');
const themeToggle = document.getElementById('theme-toggle');

// Theme Toggle (from main.css)
if (themeToggle) {
    // Check for saved theme preference or default to 'light'
    const currentTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    updateThemeIcons(currentTheme);

    themeToggle.addEventListener('click', () => {
        const theme = document.documentElement.getAttribute('data-theme');
        const newTheme = theme === 'light' ? 'dark' : 'light';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcons(newTheme);
    });
}

function updateThemeIcons(theme) {
    const sunIcon = document.getElementById('sun-icon');
    const moonIcon = document.getElementById('moon-icon');

    if (sunIcon && moonIcon) {
        if (theme === 'dark') {
            sunIcon.classList.add('hidden');
            moonIcon.classList.remove('hidden');
        } else {
            sunIcon.classList.remove('hidden');
            moonIcon.classList.add('hidden');
        }
    }
}

// Show status message
function showStatus(message, type = 'info') {
    if (!loginStatus) return;

    loginStatus.textContent = message;
    loginStatus.className = `status-message ${type}`;
    loginStatus.classList.remove('hidden');
}

// Hide status message
function hideStatus() {
    if (!loginStatus) {
        return;
    }
    loginStatus.classList.add('hidden');
}

// Handle Microsoft Login
async function handleMicrosoftLogin() {
    try {
        showStatus('Redirecting to Microsoft login...', 'info');
        microsoftLoginBtn.disabled = true;

        // Make request to backend to get Azure AD login URL
        const response = await fetch(`${API_BASE_URL}/auth/login`, {
            method: 'GET',
            credentials: 'include' // Important for cookies
        });

        if (!response.ok) {
            throw new Error('Failed to initiate login');
        }

        const data = await response.json();

        if (data.auth_url) {
            // Redirect to Azure AD login page
            window.location.href = data.auth_url;
        } else {
            throw new Error('No auth URL returned');
        }

    } catch (error) {
        console.error('Login error:', error);
        showStatus('Login failed. Please try again.', 'error');
        microsoftLoginBtn.disabled = false;
    }
}

// Check for OAuth callback parameters
function handleOAuthCallback() {
    const urlParams = new URLSearchParams(window.location.search);
    const error = urlParams.get('error');
    const errorDescription = urlParams.get('error_description');

    if (error) {
        showStatus(`Login failed: ${errorDescription || error}`, 'error');
        // Clean up URL
        window.history.replaceState({}, document.title, window.location.pathname);
    }
}

// Check if user is already authenticated
async function checkAuth() {
    try {
        const response = await fetch(`${API_BASE_URL}/auth/me`, {
            method: 'GET',
            credentials: 'include'
        });

        if (response.ok) {
            const data = await response.json();

            // User is authenticated, redirect to main app
            if (data.authenticated) {
                showStatus('Already logged in. Redirecting...', 'success');
                setTimeout(() => {
                    window.location.href = '/app';
                }, 1000);
                return true;
            }
        }

        return false;

    } catch (error) {
        console.error('Auth check error:', error);
        return false;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    // Handle OAuth callback if present
    handleOAuthCallback();

    // Only auto-redirect on login page, not on admin or other pages
    const isLoginPage = window.location.pathname.includes('/login') ||
                        window.location.pathname === '/' ||
                        window.location.pathname === '/pages/login.html';

    if (isLoginPage) {
        // Check if already authenticated and redirect to app if so
        const isAuthenticated = await checkAuth();

        if (!isAuthenticated && microsoftLoginBtn) {
            // Setup login button event listener
            microsoftLoginBtn.addEventListener('click', handleMicrosoftLogin);
        }
    } else {
        // On non-login pages (like admin), just setup the button if present
        if (microsoftLoginBtn) {
            microsoftLoginBtn.addEventListener('click', handleMicrosoftLogin);
        }
    }
});

// Utility function to get auth token from cookie or localStorage
function getAuthToken() {
    // Try to get from localStorage first
    const localToken = localStorage.getItem('session_token');
    if (localToken) {
        return localToken;
    }

    // Try to get from cookie
    const cookies = document.cookie.split('; ');
    const sessionCookie = cookies.find(row => row.startsWith('slide_review_session='));
    if (sessionCookie) {
        return sessionCookie.split('=')[1];
    }

    return null;
}

// Export for use in other scripts
window.auth = {
    checkAuth,
    showStatus,
    hideStatus,
    getAuthToken
};

// Make getAuthToken available globally for inline scripts
window.getAuthToken = getAuthToken;
