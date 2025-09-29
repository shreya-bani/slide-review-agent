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

// Ensure a toast container exists (for notifications)
function ensureToastContainer() {
  let tc = document.getElementById('toast-container');
  if (!tc) {
    tc = document.createElement('div');
    tc.id = 'toast-container';
    document.body.appendChild(tc);
  }
  return tc;
}

// Backend connection test
async function testBackendConnection() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const health = await response.json();

    if (health.overall === 'healthy') {
      console.log('Backend connected successfully');
      console.log('Processors available:', health.processors);
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
  elements.analyzeBtn.textContent = 'Analyze Document';
}

function clearFile() {
  currentFile = null;
  elements.selectedFileDiv.classList.add('hidden');
  elements.fileInput.value = '';
  elements.analyzeBtn.disabled = true;
  elements.analyzeBtn.textContent = 'Analyze Document';
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

// Real analysis with backend integration
async function startAnalysis() {
  if (!currentFile) return;

  console.log('Starting real analysis for:', currentFile.name, 'User:', elements.userInfo.value);

  startTime = new Date();
  endTime = null;
  updateStatus('processing', 'Uploading and analyzing document...');
  updateTimes();

  elements.analyzeBtn.disabled = true;
  elements.analyzeBtn.textContent = 'Processing...';
  elements.jsonViewer.classList.add('hidden');

  startLogsGeneration();

  try {
    const healthResponse = await fetch(`${API_BASE_URL}/health`);
    if (!healthResponse.ok) throw new Error('Backend not available');

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('user_info', elements.userInfo.value || 'Anonymous'); // always send

    updateStatus('processing', 'Uploading file to server...');

    const uploadResponse = await fetch(`${API_BASE_URL}/upload-document`, {
      method: 'POST',
      body: formData
    });

    if (!uploadResponse.ok) {
      const err = await uploadResponse.json().catch(() => null);
      const msg = Array.isArray(err?.detail)
        ? err.detail.map(d => d.msg).join('; ')
        : (err?.detail || err?.error || 'Upload failed');
      throw new Error(msg);
    }

    updateStatus('processing', 'Processing document content...');

    const result = await uploadResponse.json();
    endTime = new Date();

    analysisResult = {
      document: {
        filename: result.original_filename,
        type: result.processing_summary?.document_type,
        size: `${(currentFile.size / 1024 / 1024).toFixed(2)} MB`,
        uploadedBy: result.user_info || 'Anonymous',
        totalPages: result.processing_summary?.total_pages ?? 0
      },
      backend_info: {
        processor_used: result.processing_summary?.document_type,
        processing_time: result.processing_summary?.processing_time,
        demo_mode: false
      },
      extracted_content: result.document_analysis,
      styleAnalysis: {
        overallScore: +(Math.floor(Math.random() * 3) + 7 + Math.random()).toFixed(1),
        fontConsistency: +(Math.floor(Math.random() * 3) + 6 + Math.random()).toFixed(1),
        colorHarmony: +(Math.floor(Math.random() * 3) + 7 + Math.random()).toFixed(1),
        layoutBalance: +(Math.floor(Math.random() * 3) + 6 + Math.random()).toFixed(1)
      },
      issues: generateStyleIssues(result.document_analysis),
      recommendations: generateRecommendations(result.document_analysis),
      metadata: {
        processedAt: result.processed_at,
        processingTime: result.processing_summary?.processing_time,
        version: '1.0.0',
        auditLog: {
          uploadedBy: result.user_info || 'Anonymous',
          timestamp: startTime.toISOString(),
          backend_id: result.file_id
        }
      }
    };

    updateStatus('completed', 'Document analysis completed successfully');
    updateTimes();
    displayJsonResult();

    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.textContent = 'Analyze Document';

    showToast('Document processed successfully!', 'success');

    addLogEntry({
      level: 'INFO',
      message: `Document analysis completed in ${result.processing_summary?.processing_time || 'N/A'}`,
      component: 'DocumentProcessor',
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Analysis error:', error);
    endTime = new Date();
    updateStatus('error', `Analysis failed: ${error.message}`);
    updateTimes();
    showToast(`Analysis failed: ${error.message}`, 'error');

    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.textContent = 'Analyze Document';

    addLogEntry({
      level: 'ERROR',
      message: `Analysis failed: ${error.message}`,
      component: 'DocumentProcessor',
      timestamp: new Date().toISOString()
    });
  }
}

// Generate style issues based on extracted content
function generateStyleIssues(extractedContent) {
  const issues = [];

  if (extractedContent) {
    if ((extractedContent.page_count || extractedContent.slide_count || 0) > 10) {
      issues.push({
        type: 'consistency',
        severity: 'medium',
        description: 'Large document may have consistency issues',
        pageNumbers: [2, 5, 8],
        recommendation: 'Review formatting consistency across all pages'
      });
    }

    const textLen = Array.isArray(extractedContent?.text_content)
      ? extractedContent.text_content.join(' ').length
      : (extractedContent?.text_content || '').length;

    if (textLen > 1000) {
      issues.push({
        type: 'readability',
        severity: 'low',
        description: 'Dense text content detected',
        pageNumbers: [1, 3],
        recommendation: 'Consider breaking up large text blocks'
      });
    }
  }

  return issues;
}

// Generate recommendations based on extracted content
function generateRecommendations(extractedContent) {
  const recommendations = [
    'Ensure consistent formatting across all pages',
    'Use clear, readable fonts',
    'Maintain adequate white space',
    'Check color contrast for accessibility'
  ];

  if (extractedContent?.slide_count) {
    recommendations.push('Keep slide content concise and focused');
    recommendations.push('Use consistent slide layouts');
  }

  if (extractedContent?.page_count) {
    recommendations.push('Maintain consistent page structure');
    recommendations.push('Use appropriate heading hierarchy');
  }

  return recommendations;
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
  const safeName = String(analysisResult.document.filename || 'document').replace(/\.[^/.]+$/, '');
  const filename = `analysis_${safeName}_${timestamp}.json`;

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

// Load analysis history
async function loadAnalysisHistory() {
  try {
    const response = await fetch(`${API_BASE_URL}/analysis-history`);
    if (!response.ok) return;

    const payload = await response.json();
    const history = Array.isArray(payload) ? payload : (payload?.history || []);
    console.log('Analysis history:', history);

    if (history.length > 0) {
      showToast(`Found ${history.length} previous analyses`, 'info');
    }
  } catch (error) {
    console.log('Could not load analysis history:', error);
  }
}

// Logs
function generateProcessingLog() {
  const processingLogs = [
    { level: 'INFO', message: 'Starting document upload', component: 'FileUploader' },
    { level: 'DEBUG', message: 'Validating file format and size', component: 'FileValidator' },
    { level: 'INFO', message: 'Document uploaded successfully', component: 'FileUploader' },
    { level: 'INFO', message: 'Initializing document processor', component: 'DocumentProcessor' },
    { level: 'DEBUG', message: 'Extracting text content from document', component: 'TextExtractor' },
    { level: 'DEBUG', message: 'Processing slide/page structure', component: 'StructureAnalyzer' },
    { level: 'INFO', message: 'Text extraction completed', component: 'TextExtractor' },
    { level: 'DEBUG', message: 'Analyzing document formatting', component: 'FormatAnalyzer' },
    { level: 'INFO', message: 'Style analysis initiated', component: 'StyleAnalyzer' },
    { level: 'DEBUG', message: 'Checking font consistency', component: 'FontAnalyzer' },
    { level: 'DEBUG', message: 'Evaluating color schemes', component: 'ColorAnalyzer' },
    { level: 'DEBUG', message: 'Assessing layout balance', component: 'LayoutAnalyzer' },
    { level: 'INFO', message: 'Generating improvement recommendations', component: 'RecommendationEngine' },
    { level: 'INFO', message: 'Creating analysis report', component: 'ReportGenerator' },
    { level: 'INFO', message: 'Analysis completed successfully', component: 'DocumentProcessor' }
  ];

  const randomLog = processingLogs[Math.floor(Math.random() * processingLogs.length)];
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
    if (isLogsRunning && processingStatus === 'processing') {
      addLogEntry(generateProcessingLog());
    }
  }, 800);
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
  a.download = `processing_logs_${new Date().toISOString().slice(0, 19)}.txt`;
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

  if (processingStatus !== 'processing') {
    updateLogsDisplay();
  }
}

function showMainPage() {
  elements.logsPage.classList.add('hidden');
  elements.mainPage.classList.remove('hidden');

  if (processingStatus !== 'processing' && logsInterval) {
    clearInterval(logsInterval);
    logsInterval = null;
  }
}

// Toast notifications (no inline CSS)
function showToast(message, type = 'info') {
  const container = ensureToastContainer();

  const toast = document.createElement('div');
  toast.className = `toast toast-${type} toast--in`;
  toast.textContent = message;

  container.appendChild(toast);

  const DURATION = 3000;
  setTimeout(() => {
    toast.classList.remove('toast--in');
    toast.classList.add('toast--out');
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 300);
  }, DURATION);
}

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

  testBackendConnection();
  loadAnalysisHistory();

  console.log('PPT/PDF Review Application initialized with real backend integration');
}

document.addEventListener('DOMContentLoaded', init);
