// Backend API base URL
const API_BASE_URL = 'http://localhost:8000';

// Application state
let currentFile = null;
let analysisResult = null;
let processingStatus = 'idle';
let startTime = null;
let endTime = null;
let logs = [];
let isLogsRunning = true;
let autoScroll = true;
let logsInterval = null;

// DOM elements
const elements = {
    // Main page
    mainPage: document.getElementById('main-page'),
    logsPage: document.getElementById('logs-page'),
    
    // File upload
    fileUploadArea: document.getElementById('file-upload-area'),
    browseBtn: document.getElementById('browse-btn'),
    fileInput: document.getElementById('file-input'),
    selectedFileDiv: document.getElementById('selected-file'),
    fileName: document.getElementById('file-name'),
    fileSize: document.getElementById('file-size'),
    clearFileBtn: document.getElementById('clear-file'),
    userInfo: document.getElementById('user-info'),
    analyzeBtn: document.getElementById('analyze-btn'),
    
    // Status
    statusLine: document.getElementById('status-line'),
    statusMessage: document.getElementById('status-message'),
    startTimeEl: document.getElementById('start-time'),
    endTimeEl: document.getElementById('end-time'),
    durationEl: document.getElementById('duration'),
    
    // JSON viewer
    jsonViewer: document.getElementById('json-viewer'),
    jsonDisplay: document.getElementById('json-display'),
    copyJsonBtn: document.getElementById('copy-json'),
    downloadJsonBtn: document.getElementById('download-json'),
    copyIcon: document.getElementById('copy-icon'),
    checkIcon: document.getElementById('check-icon'),
    copyText: document.getElementById('copy-text'),
    
    // Navigation
    logsBtn: document.getElementById('logs-btn'),
    backBtn: document.getElementById('back-btn'),
    themeToggle: document.getElementById('theme-toggle'),
    themeToggleLogs: document.getElementById('theme-toggle-logs'),
    
    // Logs page
    logsContent: document.getElementById('logs-content'),
    logsCount: document.getElementById('logs-count'),
    logsCountBadge: document.getElementById('logs-count-badge'),
    logsStatus: document.getElementById('logs-status'),
    toggleLogsBtn: document.getElementById('toggle-logs'),
    autoScrollToggle: document.getElementById('auto-scroll-toggle'),
    downloadLogsBtn: document.getElementById('download-logs'),
    clearLogsBtn: document.getElementById('clear-logs'),
    pauseIcon: document.getElementById('pause-icon'),
    playIcon: document.getElementById('play-icon')
};

// Backend connection test
async function testBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const health = await response.json();
        
        if (health.overall === 'healthy') {
            console.log('Backend connected successfully');
            showToast('Backend connected successfully', 'success');
        } else {
            console.log('Backend partially healthy:', health);
            showToast('Backend partially healthy - check configuration', 'warning');
        }
    } catch (error) {
        console.error('Backend connection failed:', error);
        showToast('Backend connection failed - make sure server is running', 'error');
    }
}

// Theme management
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.className = savedTheme === 'dark' ? 'dark' : '';
    updateThemeIcons();
}

function toggleTheme() {
    const isDark = document.documentElement.classList.contains('dark');
    document.documentElement.className = isDark ? '' : 'dark';
    localStorage.setItem('theme', isDark ? 'light' : 'dark');
    updateThemeIcons();
}

function updateThemeIcons() {
    const isDark = document.documentElement.classList.contains('dark');
    const sunIcons = document.querySelectorAll('#sun-icon, #sun-icon-logs');
    const moonIcons = document.querySelectorAll('#moon-icon, #moon-icon-logs');
    
    sunIcons.forEach(icon => icon.classList.toggle('hidden', isDark));
    moonIcons.forEach(icon => icon.classList.toggle('hidden', !isDark));
}

// File handling
function handleFileSelect(file) {
    if (!file || (!file.name.endsWith('.pptx') && !file.name.endsWith('.pdf'))) {
        showToast('Please select a .pptx or .pdf file', 'error');
        return;
    }
    
    currentFile = file;
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    elements.selectedFileDiv.classList.remove('hidden');
    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.textContent = 'See Entire Analysis Running';
}

function clearFile() {
    currentFile = null;
    elements.selectedFileDiv.classList.add('hidden');
    elements.fileInput.value = '';
    elements.analyzeBtn.disabled = true;
    elements.analyzeBtn.textContent = 'See Entire Analysis Running';
}

// Drag and drop
function setupDragAndDrop() {
    elements.fileUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.fileUploadArea.classList.add('drag-over');
    });
    
    elements.fileUploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        elements.fileUploadArea.classList.remove('drag-over');
    });
    
    elements.fileUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.fileUploadArea.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
}

// Status management
function updateStatus(status, message = '') {
    processingStatus = status;
    const statusBadge = elements.statusLine.querySelector('.badge');
    const statusIcon = elements.statusLine.querySelector('.status-icon');
    
    // Clear all status classes
    statusBadge.className = 'badge';
    
    switch (status) {
        case 'processing':
            statusBadge.classList.add('badge-warning');
            statusBadge.textContent = 'Processing';
            statusIcon.classList.add('loading-spinner');
            break;
        case 'completed':
            statusBadge.classList.add('badge-success');
            statusBadge.textContent = 'Completed';
            statusIcon.classList.remove('loading-spinner');
            break;
        case 'error':
            statusBadge.classList.add('badge-error');
            statusBadge.textContent = 'Error';
            statusIcon.classList.remove('loading-spinner');
            break;
        default:
            statusBadge.classList.add('badge-outline');
            statusBadge.textContent = 'Ready';
            statusIcon.classList.remove('loading-spinner');
    }
    
    elements.statusMessage.textContent = message;
}

function updateTimes() {
    const formatTime = (date) => {
        if (!date) return '--:--:--';
        return date.toLocaleTimeString('en-US', { hour12: false });
    };
    
    elements.startTimeEl.textContent = formatTime(startTime);
    elements.endTimeEl.textContent = formatTime(endTime);
    
    if (startTime && endTime) {
        const duration = (endTime.getTime() - startTime.getTime()) / 1000;
        elements.durationEl.textContent = `${duration.toFixed(1)}s`;
    } else {
        elements.durationEl.textContent = '--';
    }
}

// Analysis with backend integration
async function startAnalysis() {
    if (!currentFile) return;
    
    console.log('Starting analysis for:', currentFile.name, 'User:', elements.userInfo.value);
    
    startTime = new Date();
    endTime = null;
    updateStatus('processing', 'Analyzing document content and professional styling...');
    updateTimes();
    
    elements.analyzeBtn.disabled = true;
    elements.analyzeBtn.textContent = 'Analysis Running...';
    elements.jsonViewer.classList.add('hidden');
    
    try {
        // Test backend connection first
        const healthResponse = await fetch(`${API_BASE_URL}/health`);
        if (!healthResponse.ok) {
            throw new Error('Backend not available');
        }
        
        // For now, simulate the analysis (later we'll add real file upload)
        // TODO: Replace with actual file upload and processing
        
        setTimeout(async () => {
            endTime = new Date();
            
            // Call backend for analysis (placeholder for now)
            try {
                const configResponse = await fetch(`${API_BASE_URL}/config`);
                const config = await configResponse.json();
                
                // Generate mock analysis result with backend info
                analysisResult = {
                    document: {
                        filename: currentFile.name,
                        type: currentFile.name.endsWith('.pptx') ? 'PowerPoint' : 'PDF',
                        size: `${(currentFile.size / 1024 / 1024).toFixed(2)} MB`,
                        uploadedBy: elements.userInfo.value || 'Anonymous',
                        totalPages: currentFile.name.endsWith('.pptx') ? 
                            Math.floor(Math.random() * 20) + 5 : 
                            Math.floor(Math.random() * 50) + 10
                    },
                    backend_info: {
                        llm_provider: config.llm_provider,
                        llm_model: config.llm_model,
                        demo_mode: config.demo_mode
                    },
                    styleAnalysis: {
                        overallScore: +(Math.floor(Math.random() * 3) + 7 + Math.random()).toFixed(1),
                        fontConsistency: +(Math.floor(Math.random() * 3) + 6 + Math.random()).toFixed(1),
                        colorHarmony: +(Math.floor(Math.random() * 3) + 7 + Math.random()).toFixed(1),
                        layoutBalance: +(Math.floor(Math.random() * 3) + 6 + Math.random()).toFixed(1)
                    },
                    issues: [
                        {
                            type: 'typography',
                            severity: 'medium',
                            description: 'Inconsistent font sizes detected',
                            pageNumbers: [2, 5, 8],
                            recommendation: 'Use consistent heading hierarchy'
                        },
                        {
                            type: 'color_accessibility',
                            severity: 'low',
                            description: 'Some text has low contrast ratio',
                            pageNumbers: [3, 7],
                            recommendation: 'Increase contrast for better readability'
                        }
                    ],
                    recommendations: [
                        'Standardize font hierarchy across all slides/pages',
                        'Improve color contrast for accessibility',
                        'Add consistent spacing between elements',
                        'Use professional color palette',
                        'Ensure all text is readable against backgrounds'
                    ],
                    metadata: {
                        processedAt: new Date().toISOString(),
                        processingTime: `${((endTime.getTime() - startTime.getTime()) / 1000).toFixed(1)}s`,
                        version: '1.0.0',
                        auditLog: {
                            uploadedBy: elements.userInfo.value || 'Anonymous',
                            timestamp: startTime.toISOString()
                        }
                    }
                };
                
                updateStatus('completed');
                updateTimes();
                displayJsonResult();
                
                elements.analyzeBtn.disabled = false;
                elements.analyzeBtn.textContent = 'See Entire Analysis Running';
                
                showToast('Analysis completed successfully! (Connected to backend)', 'success');
                
            } catch (backendError) {
                console.error('Backend error:', backendError);
                updateStatus('error', 'Backend connection failed');
                showToast('Backend connection failed: ' + backendError.message, 'error');
            }
        }, 3500);
        
    } catch (error) {
        console.error('Analysis error:', error);
        updateStatus('error', 'Analysis failed');
        showToast('Analysis failed: ' + error.message, 'error');
        
        elements.analyzeBtn.disabled = false;
        elements.analyzeBtn.textContent = 'See Entire Analysis Running';
    }
}

// JSON display
function displayJsonResult() {
    if (!analysisResult) return;
    
    elements.jsonDisplay.textContent = JSON.stringify(analysisResult, null, 2);
    elements.jsonViewer.classList.remove('hidden');
}

// Copy and download functions
async function copyToClipboard() {
    if (!analysisResult) return;
    
    try {
        await navigator.clipboard.writeText(JSON.stringify(analysisResult, null, 2));
        
        elements.copyIcon.classList.add('hidden');
        elements.checkIcon.classList.remove('hidden');
        elements.copyText.textContent = 'Copied';
        
        setTimeout(() => {
            elements.copyIcon.classList.remove('hidden');
            elements.checkIcon.classList.add('hidden');
            elements.copyText.textContent = 'Copy JSON';
        }, 2000);
        
        showToast('JSON copied to clipboard!', 'success');
    } catch (error) {
        showToast('Failed to copy to clipboard', 'error');
    }
}

function downloadJson() {
    if (!analysisResult) return;
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `report_${analysisResult.document.filename.replace(/\.[^/.]+$/, '')}_${timestamp}.json`;
    
    const blob = new Blob([JSON.stringify(analysisResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast(`${filename} downloaded successfully!`, 'success');
}

// Logs management
function generateMockLog() {
    const mockLogs = [
        { level: 'INFO', message: 'Document processor initialized', component: 'DocumentProcessor' },
        { level: 'DEBUG', message: 'Extracting text from slide 1 of 12', component: 'TextExtractor' },
        { level: 'INFO', message: 'Style analysis started', component: 'StyleAnalyzer' },
        { level: 'DEBUG', message: 'Analyzing font consistency across document', component: 'FontAnalyzer' },
        { level: 'WARN', message: 'Low contrast detected on slide 3', component: 'ColorAnalyzer' },
        { level: 'INFO', message: 'Layout balance analysis complete', component: 'LayoutAnalyzer' },
        { level: 'DEBUG', message: 'Generating recommendations based on findings', component: 'RecommendationEngine' },
        { level: 'INFO', message: 'Analysis complete. Report generated successfully', component: 'DocumentProcessor' },
        { level: 'ERROR', message: 'Failed to process image on slide 7', component: 'ImageProcessor' },
        { level: 'INFO', message: 'Audit log entry created', component: 'AuditLogger' }
    ];
    
    const randomLog = mockLogs[Math.floor(Math.random() * mockLogs.length)];
    return {
        ...randomLog,
        timestamp: new Date().toISOString()
    };
}

function addLogEntry(logEntry) {
    logs.push(logEntry);
    updateLogsDisplay();
    updateLogsCount();
    
    if (autoScroll) {
        setTimeout(() => {
            elements.logsContent.scrollTop = elements.logsContent.scrollHeight;
        }, 10);
    }
}

function updateLogsDisplay() {
    if (logs.length === 0) {
        elements.logsContent.innerHTML = '<div class="logs-empty">No logs yet. Logs will appear here when processing starts.</div>';
        return;
    }
    
    const logsHtml = logs.map(log => `
        <div class="log-entry">
            <span class="log-timestamp">${new Date(log.timestamp).toLocaleTimeString()}</span>
            <span class="log-level log-level-${log.level.toLowerCase()}">${log.level}</span>
            ${log.component ? `<span class="log-component">[${log.component}]</span>` : ''}
            <span class="log-message">${log.message}</span>
        </div>
    `).join('');
    
    elements.logsContent.innerHTML = logsHtml;
}

function updateLogsCount() {
    const count = logs.length;
    elements.logsCount.textContent = `${count} log entries`;
    elements.logsCountBadge.textContent = `${count} entries`;
}

function startLogsGeneration() {
    if (logsInterval) clearInterval(logsInterval);
    
    logsInterval = setInterval(() => {
        if (isLogsRunning) {
            addLogEntry(generateMockLog());
        }
    }, 1200);
}

function toggleLogs() {
    isLogsRunning = !isLogsRunning;
    
    elements.logsStatus.textContent = isLogsRunning ? 'Running' : 'Paused';
    elements.logsStatus.className = isLogsRunning ? 'badge badge-primary' : 'badge badge-secondary';
    
    elements.pauseIcon.classList.toggle('hidden', !isLogsRunning);
    elements.playIcon.classList.toggle('hidden', isLogsRunning);
}

function toggleAutoScroll() {
    autoScroll = !autoScroll;
    elements.autoScrollToggle.textContent = `Auto-scroll: ${autoScroll ? 'On' : 'Off'}`;
}

function clearLogs() {
    logs = [];
    updateLogsDisplay();
    updateLogsCount();
}

function downloadLogs() {
    if (logs.length === 0) return;
    
    const logText = logs.map(log => 
        `[${log.timestamp}] ${log.level} ${log.component ? `[${log.component}]` : ''} ${log.message}`
    ).join('\n');
    
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `terminal_logs_${new Date().toISOString().slice(0, 19)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast('Logs downloaded successfully!', 'success');
}

// Navigation
function showLogsPage() {
    elements.mainPage.classList.add('hidden');
    elements.logsPage.classList.remove('hidden');
    startLogsGeneration();
}

function showMainPage() {
    elements.logsPage.classList.add('hidden');
    elements.mainPage.classList.remove('hidden');
    if (logsInterval) {
        clearInterval(logsInterval);
        logsInterval = null;
    }
}

// Toast notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 16px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        color: var(--foreground);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    if (type === 'success') {
        toast.style.borderColor = 'var(--success)';
        toast.style.color = 'var(--success)';
    } else if (type === 'error') {
        toast.style.borderColor = 'var(--error)';
        toast.style.color = 'var(--error)';
    }
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 3000);
}

// Add CSS for toast animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Event listeners
function setupEventListeners() {
    // File upload
    elements.fileUploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.browseBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFileSelect(file);
    });
    elements.clearFileBtn.addEventListener('click', clearFile);
    elements.analyzeBtn.addEventListener('click', startAnalysis);
    
    // JSON viewer
    elements.copyJsonBtn.addEventListener('click', copyToClipboard);
    elements.downloadJsonBtn.addEventListener('click', downloadJson);
    
    // Navigation
    elements.logsBtn.addEventListener('click', showLogsPage);
    elements.backBtn.addEventListener('click', showMainPage);
    elements.themeToggle.addEventListener('click', toggleTheme);
    elements.themeToggleLogs.addEventListener('click', toggleTheme);
    
    // Logs controls
    elements.toggleLogsBtn.addEventListener('click', toggleLogs);
    elements.autoScrollToggle.addEventListener('click', toggleAutoScroll);
    elements.clearLogsBtn.addEventListener('click', clearLogs);
    elements.downloadLogsBtn.addEventListener('click', downloadLogs);
}

// Initialize application
function init() {
    initTheme();
    setupEventListeners();
    setupDragAndDrop();
    updateStatus('idle');
    updateTimes();
    updateLogsCount();
    
    // Test backend connection
    testBackendConnection();
    
    console.log('PPT/PDF Review Application initialized with backend connection');
}

// Start the application
document.addEventListener('DOMContentLoaded', init);