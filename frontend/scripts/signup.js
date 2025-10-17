// Theme toggle
const themeToggle = document.getElementById('theme-toggle');
const sunIcon = document.getElementById('sun-icon');
const moonIcon = document.getElementById('moon-icon');
const htmlElement = document.documentElement;

// Check for saved theme preference or default to light mode
const currentTheme = localStorage.getItem('theme') || 'light';
if (currentTheme === 'dark') {
    htmlElement.classList.add('dark');
    sunIcon.classList.add('hidden');
    moonIcon.classList.remove('hidden');
}

themeToggle.addEventListener('click', () => {
    htmlElement.classList.toggle('dark');
    const isDark = htmlElement.classList.contains('dark');

    sunIcon.classList.toggle('hidden', isDark);
    moonIcon.classList.toggle('hidden', !isDark);

    localStorage.setItem('theme', isDark ? 'dark' : 'light');
});

// Password toggle for main password field
const togglePassword = document.getElementById('toggle-password');
const passwordInput = document.getElementById('password');
const eyeIcon = document.getElementById('eye-icon');
const eyeOffIcon = document.getElementById('eye-off-icon');

togglePassword.addEventListener('click', () => {
    const type = passwordInput.type === 'password' ? 'text' : 'password';
    passwordInput.type = type;

    eyeIcon.classList.toggle('hidden');
    eyeOffIcon.classList.toggle('hidden');
});

// Password toggle for confirm password field
const toggleConfirmPassword = document.getElementById('toggle-confirm-password');
const confirmPasswordInput = document.getElementById('confirm-password');
const eyeIconConfirm = document.getElementById('eye-icon-confirm');
const eyeOffIconConfirm = document.getElementById('eye-off-icon-confirm');

toggleConfirmPassword.addEventListener('click', () => {
    const type = confirmPasswordInput.type === 'password' ? 'text' : 'password';
    confirmPasswordInput.type = type;

    eyeIconConfirm.classList.toggle('hidden');
    eyeOffIconConfirm.classList.toggle('hidden');
});

// Form validation and submission
const signupForm = document.getElementById('signup-form');
const errorMessage = document.getElementById('error-message');

signupForm.addEventListener('submit', (e) => {
    e.preventDefault();

    const fullName = document.getElementById('full-name').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm-password').value;
    const agreeTerms = document.getElementById('agree-terms').checked;

    // Validate passwords match
    if (password !== confirmPassword) {
        errorMessage.textContent = 'Passwords do not match';
        errorMessage.classList.add('show');
        setTimeout(() => errorMessage.classList.remove('show'), 3000);
        return;
    }

    // Validate password length
    if (password.length < 8) {
        errorMessage.textContent = 'Password must be at least 8 characters long';
        errorMessage.classList.add('show');
        setTimeout(() => errorMessage.classList.remove('show'), 3000);
        return;
    }

    // Validate terms agreement
    if (!agreeTerms) {
        errorMessage.textContent = 'You must agree to the Terms of Service and Privacy Policy';
        errorMessage.classList.add('show');
        setTimeout(() => errorMessage.classList.remove('show'), 3000);
        return;
    }

    // Add your signup logic here
    console.log('Signup attempt:', { fullName, email, password, agreeTerms });

    // Example: Show error for demo (remove in production)
    // errorMessage.classList.add('show');
    // setTimeout(() => errorMessage.classList.remove('show'), 3000);

    // Example: Successful signup redirect
    // window.location.href = 'login.html';

    // Example: API call
    /*
    fetch('/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fullName, email, password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Redirect to login page with success message
            window.location.href = 'login.html?registered=true';
        } else {
            errorMessage.textContent = data.message || 'Registration failed';
            errorMessage.classList.add('show');
            setTimeout(() => errorMessage.classList.remove('show'), 3000);
        }
    })
    .catch(error => {
        errorMessage.textContent = 'An error occurred. Please try again.';
        errorMessage.classList.add('show');
        setTimeout(() => errorMessage.classList.remove('show'), 3000);
    });
    */
});

// Password strength indicator (optional enhancement)
passwordInput.addEventListener('input', () => {
    const password = passwordInput.value;
    // You can add password strength indicator logic here
    // For example, checking for uppercase, lowercase, numbers, special chars
});
