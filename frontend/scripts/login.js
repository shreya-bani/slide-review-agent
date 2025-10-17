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

// Password toggle
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

// Form submission
const loginForm = document.getElementById('login-form');
const errorMessage = document.getElementById('error-message');

loginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const remember = document.getElementById('remember').checked;
    
    // Add your login logic here
    console.log('Login attempt:', { email, password, remember });
    
    // Example: Show error for demo (remove in production)
    // errorMessage.classList.add('show');
    // setTimeout(() => errorMessage.classList.remove('show'), 3000);
    
    // Example: Successful login redirect
    // window.location.href = 'main.html';
    
    // Example: API call
    /*
    fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, remember })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            window.location.href = 'main.html';
        } else {
            errorMessage.textContent = data.message || 'Invalid credentials';
            errorMessage.classList.add('show');
        }
    })
    .catch(error => {
        errorMessage.textContent = 'An error occurred. Please try again.';
        errorMessage.classList.add('show');
    });
    */
});