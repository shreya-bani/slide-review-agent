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
let currentView = 'visual';
let filteredFindings = [];

// DOM elements
const elements = {
  mainPage: document.getElementById('main-page'),
  logsPage: document.getElementById('logs-page'),
  fileUploadArea: document.getElementById('file-upload-area'),
  browseBtn: document.getElementById('browse-btn'),
  selectedFileDiv: document.getElementById('selected-file'),
  fileName: document.getElementById('file-name'),
  fileSize: document.getElementById('file-size'),
  clearFileBtn: document.getElementById('clear-file'),
  statusLine: document.getElementById('status-line'),
  statusMessage: document.getElementById('status-message'),
  startTimeEl: document.getElementById('start-time'),
  endTimeEl: document.getElementById('end-time'),
  durationEl: document.getElementById('duration'),
  visualReport: document.getElementById('visual-report'),
  visualViewBtn: document.getElementById('visual-view-btn'),
  jsonViewBtn: document.getElementById('json-view-btn'),
  visualContent: document.getElementById('visual-content'),
  jsonContent: document.getElementById('json-content'),
  jsonDisplay: document.getElementById('json-display'),
  copyJsonBtn: document.getElementById('copy-json'),
  downloadJsonBtn: document.getElementById('download-json'),
  exportReportBtn: document.getElementById('export-report-btn'),
  copyIcon: document.getElementById('copy-icon'),
  checkIcon: document.getElementById('check-icon'),
  copyText: document.getElementById('copy-text'),
  logsBtn: document.getElementById('logs-btn'),
  backBtn: document.getElementById('back-btn'),
  themeToggle: document.getElementById('theme-toggle'),
  themeToggleLogs: document.getElementById('theme-toggle-logs'),
  logsContent: document.getElementById('logs-content'),
  logsCount: document.getElementById('logs-count'),
  logsCountBadge: document.getElementById('logs-count-badge'),
  logsStatus: document.getElementById('logs-status'),
  toggleLogsBtn: document.getElementById('toggle-logs'),
  autoScrollToggle: document.getElementById('auto-scroll-toggle'),
  downloadLogsBtn: document.getElementById('download-logs'),
  clearLogsBtn: document.getElementById('clear-logs'),
  pauseIcon: document.getElementById('pause-icon'),
  playIcon: document.getElementById('play-icon'),
  severityFilter: document.getElementById('severity-filter'),
  categoryFilter: document.getElementById('category-filter'),
  fileInput: document.getElementById('file-input'),
  analyzeBtn: document.getElementById('analyze-btn'),
  userInfo: document.getElementById('user-info'),
  metaTitle: document.getElementById('meta-title'), 
};

function updateAnalyzeEnabled() {
  const hasFile = !!elements.fileInput?.files?.[0];
  const hasName = !!elements.userInfo?.value.trim();
  elements.analyzeBtn.disabled = !(hasFile && hasName);
}
elements.fileInput?.addEventListener('change', updateAnalyzeEnabled);
elements.userInfo?.addEventListener('input', updateAnalyzeEnabled);

function ensureToastContainer() {
  let tc = document.getElementById('toast-container');
  if (!tc) {
    tc = document.createElement('div');
    tc.id = 'toast-container';
    document.body.appendChild(tc);
  }
  return tc;
}

async function testBackendConnection() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const health = await response.json();
    if (health.overall === 'healthy') {
      console.log('Backend connected successfully');
      showToast('Backend connected successfully', 'success');
    } else {
      showToast('Backend partially healthy - check configuration', 'warning');
    }
  } catch (error) {
    console.error('Backend connection failed:', error);
    showToast('Backend connection failed - make sure server is running', 'error');
  }
}

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
  sunIcons.forEach((icon) => icon.classList.toggle('hidden', isDark));
  moonIcons.forEach((icon) => icon.classList.toggle('hidden', !isDark));
}

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
    if (files.length > 0) handleFileSelect(files[0]);
  });
}

function updateStatus(status, message = '') {
  processingStatus = status;
  const statusBadge = elements.statusLine?.querySelector('.badge');
  const statusIcon = elements.statusLine?.querySelector('.status-icon');
  if (!statusBadge || !statusIcon) return;

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
  elements.statusMessage && (elements.statusMessage.textContent = message);
}

function updateTimes() {
  const formatTime = (date) =>
    date ? date.toLocaleTimeString('en-US', { hour12: false }) : '--:--:--';
  elements.startTimeEl && (elements.startTimeEl.textContent = formatTime(startTime));
  elements.endTimeEl && (elements.endTimeEl.textContent = formatTime(endTime));
  if (elements.durationEl) {
    if (startTime && endTime) {
      const duration = (endTime.getTime() - startTime.getTime()) / 1000;
      elements.durationEl.textContent = `${duration.toFixed(1)}s`;
    } else {
      elements.durationEl.textContent = '--';
    }
  }
}

async function startAnalysis() {
  if (!currentFile) return;
  console.log('Starting analysis for:', currentFile.name, 'User:', elements.userInfo?.value);
  startTime = new Date();
  endTime = null;
  updateStatus('processing', 'Uploading and analyzing document...');
  updateTimes();
  elements.analyzeBtn.disabled = true;
  elements.analyzeBtn.textContent = 'Processing...';
  elements.visualReport?.classList.add('hidden');
  startLogsGeneration();

  try {
    const healthResponse = await fetch(`${API_BASE_URL}/health`);
    if (!healthResponse.ok) throw new Error('Backend not available');

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('user_info', elements.userInfo?.value || 'Anonymous');
    updateStatus('processing', 'Uploading file to server...');

    const uploadResponse = await fetch(`${API_BASE_URL}/upload-document`, {
      method: 'POST',
      body: formData,
    });

    if (!uploadResponse.ok) {
      const err = await uploadResponse.json().catch(() => null);
      const msg = Array.isArray(err?.detail)
        ? err.detail.map((d) => d.msg).join('; ')
        : err?.detail || err?.error || 'Upload failed';
      throw new Error(msg);
    }

    updateStatus('processing', 'Processing document content...');
    const result = await uploadResponse.json();
    endTime = new Date();

    analysisResult = result;

    updateStatus('completed', 'Document analysis completed successfully');
    updateTimes();
    displayVisualReport();
    displayJsonResult();
    elements.visualReport?.classList.remove('hidden');
    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.textContent = 'Analyze Document';
    showToast('Document processed successfully!', 'success');

    addLogEntry({
      level: 'INFO',
      message: 'Document analysis completed',
      component: 'DocumentProcessor',
      timestamp: new Date().toISOString(),
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
      timestamp: new Date().toISOString(),
    });
  }
}

function displayVisualReport() {
  if (!analysisResult) return;

  // Use analysis_summary from the backend response
  const summary = analysisResult.analysis_summary || {};
  const metadata = analysisResult.metadata || {};
  
  // Overall score (calculate from severity breakdown)
  const severity = summary.severity_breakdown || {};
  const total = summary.total_issues || 0;
  
  // Calculate score: fewer issues = higher score
  // Scale: 0 issues = 100, proportionally decrease with more issues
  const score = total === 0 ? 100 : Math.max(0, 100 - (total / 10));
  
  const scoreEl = document.getElementById('overall-score');
  if (scoreEl) scoreEl.textContent = score.toFixed(1);

  const progressCircle = document.getElementById('score-ring-progress');
  const circumference = 283;
  const offset = circumference - (score / 100) * circumference;
  if (progressCircle) progressCircle.style.strokeDashoffset = offset;

  const statusEl = document.getElementById('score-status');
  if (statusEl && progressCircle) {
    if (score >= 90) {
      statusEl.textContent = 'Excellent';
      statusEl.style.color = 'var(--success)';
      progressCircle.style.stroke = 'var(--success)';
    } else if (score >= 70) {
      statusEl.textContent = 'Good';
      statusEl.style.color = 'var(--primary)';
      progressCircle.style.stroke = 'var(--primary)';
    } else {
      statusEl.textContent = 'Needs Improvement';
      statusEl.style.color = 'var(--warning)';
      progressCircle.style.stroke = 'var(--warning)';
    }
  }

  // Severity breakdown
  const totalIssuesEl = document.getElementById('total-issues');
  if (totalIssuesEl) totalIssuesEl.textContent = `${total} issue${total !== 1 ? 's' : ''}`;
  
  const errorCount = document.getElementById('error-count');
  const warningCount = document.getElementById('warning-count');
  const infoCount = document.getElementById('info-count');
  
  if (errorCount) errorCount.textContent = severity.error || 0;
  if (warningCount) warningCount.textContent = severity.warning || 0;
  if (infoCount) infoCount.textContent = severity.info || severity.suggestion || 0;

  if (total > 0) {
    const errorBar = document.getElementById('error-bar');
    const warningBar = document.getElementById('warning-bar');
    const infoBar = document.getElementById('info-bar');
    if (errorBar) errorBar.style.width = `${((severity.error || 0) / total) * 100}%`;
    if (warningBar) warningBar.style.width = `${((severity.warning || 0) / total) * 100}%`;
    if (infoBar) infoBar.style.width = `${((severity.info || severity.suggestion || 0) / total) * 100}%`;
  }

  // Category breakdown
  const categories = summary.category_breakdown || {};
  const categoryList = document.getElementById('category-list');
  if (categoryList) {
    categoryList.innerHTML = '';
    Object.entries(categories).forEach(([name, count]) => {
      const item = document.createElement('div');
      item.className = 'category-item';
      item.innerHTML = `
        <span class="category-name">${name.replace('-', ' ')}</span>
        <span class="category-count">${count}</span>
      `;
      categoryList.appendChild(item);
    });
  }

  const coverageBadge = document.getElementById('coverage-badge');
  if (coverageBadge) {
    const contentStats = analysisResult.content_statistics || {};
    const totalPages = contentStats.total_pages || 0;
    coverageBadge.textContent = `${totalPages} pages analyzed`;
  }

  // Metadata
  const metaFilename = document.getElementById('meta-filename');
  const metaType = document.getElementById('meta-type');
  const metaTitle = document.getElementById('meta-title');
  const metaAuthor = document.getElementById('meta-author');
  const metaSlides = document.getElementById('meta-slides');
  const metaTimestamp = document.getElementById('meta-timestamp');

  if (metaFilename) metaFilename.textContent = analysisResult.original_filename || '--';
  if (metaType) metaType.textContent = metadata.document_type?.toUpperCase() || '--';
  if (metaTitle) metaTitle.textContent = metadata.title || '--';
  if (metaAuthor) metaAuthor.textContent = metadata.author || '--';
  if (metaSlides) metaSlides.textContent = analysisResult.content_statistics?.total_pages || '--';
  if (metaTimestamp) metaTimestamp.textContent = new Date(analysisResult.processed_at || Date.now()).toLocaleString();

  // Findings
  filteredFindings = analysisResult.findings || [];
  displayFindings();
  updateCategoryFilter();
}

function displayFindings() {
  const findingsTable = document.getElementById('findings-table');
  if (!findingsTable) return;

  findingsTable.innerHTML = '';

  if (filteredFindings.length === 0) {
    findingsTable.innerHTML =
      '<div style="padding: 32px; text-align: center; color: var(--muted-foreground);">No findings match the current filters.</div>';
    return;
  }

  filteredFindings.forEach((finding, index) => {
    const row = document.createElement('div');
    row.className = 'finding-row';
    
    // Use found_text for summary instead of description
    const summaryText = finding.found_text || finding.description || 'No text available';
    const truncatedText = summaryText.length > 80 ? summaryText.substring(0, 80) + '...' : summaryText;
    
    // Map severity to display correctly
    const severity = finding.severity || 'info';
    
    // Format location correctly
    const slideNum = (finding.page_or_slide_index || 0) + 1;
    const location = `Slide ${slideNum}`;
    
    row.innerHTML = `
      <div class="finding-summary" data-finding-index="${index}">
        <div class="finding-severity">
          <span class="severity-dot severity-${severity}"></span>
        </div>
        <div class="finding-id">${severity}</div>
        <div class="finding-desc">${truncatedText}</div>
        <div class="finding-location">${location}</div>
        <div class="finding-category">${finding.category || 'general'}</div>
      </div>
      <div class="finding-details">
        <div class="finding-detail-section">
          <div class="finding-detail-label">Found Text</div>
          <div class="finding-text">${finding.found_text || 'N/A'}</div>
        </div>
        <div class="finding-detail-section">
          <div class="finding-detail-label">Issue Explanation</div>
          <div class="finding-text">${finding.description || 'No description available'}</div>
        </div>
        <div class="finding-detail-section">
          <div class="finding-detail-label">Suggestion</div>
          <div class="finding-suggestion">${finding.suggestion || 'No suggestion available'}</div>
        </div>
      </div>
    `;
    findingsTable.appendChild(row);
  });

  findingsTable.querySelectorAll('.finding-summary').forEach((summary) => {
    summary.addEventListener('click', () => {
      const row = summary.parentElement;
      row.classList.toggle('expanded');
    });
  });
}

function updateCategoryFilter() {
  const categories = new Set();
  (analysisResult?.findings || []).forEach((f) => categories.add(f.category));

  const categoryFilter = elements.categoryFilter;
  if (!categoryFilter) return;

  categoryFilter.innerHTML = '<option value="all">All Categories</option>';
  categories.forEach((cat) => {
    const option = document.createElement('option');
    option.value = cat;
    option.textContent = cat.replace('-', ' ');
    categoryFilter.appendChild(option);
  });
}

function filterFindings() {
  const severityFilter = elements.severityFilter?.value || 'all';
  const categoryFilter = elements.categoryFilter?.value || 'all';

  filteredFindings = (analysisResult?.findings || []).filter((finding) => {
    const matchesSeverity = severityFilter === 'all' || finding.severity === severityFilter;
    const matchesCategory = categoryFilter === 'all' || finding.category === categoryFilter;
    return matchesSeverity && matchesCategory;
  });

  displayFindings();
}

function toggleView(view) {
  currentView = view;
  if (view === 'visual') {
    elements.visualContent?.classList.remove('hidden');
    elements.jsonContent?.classList.add('hidden');
    elements.visualViewBtn?.classList.add('active');
    elements.jsonViewBtn?.classList.remove('active');
  } else {
    elements.visualContent?.classList.add('hidden');
    elements.jsonContent?.classList.remove('hidden');
    elements.visualViewBtn?.classList.remove('active');
    elements.jsonViewBtn?.classList.add('active');
  }
}

function displayJsonResult() {
  if (!analysisResult) return;
  if (elements.jsonDisplay) {
    elements.jsonDisplay.textContent = JSON.stringify(analysisResult, null, 2);
  }
}

async function copyToClipboard() {
  if (!analysisResult) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(analysisResult, null, 2));
    elements.copyIcon?.classList.add('hidden');
    elements.checkIcon?.classList.remove('hidden');
    elements.copyText && (elements.copyText.textContent = 'Copied');
    setTimeout(() => {
      elements.copyIcon?.classList.remove('hidden');
      elements.checkIcon?.classList.add('hidden');
      elements.copyText && (elements.copyText.textContent = 'Copy JSON');
    }, 2000);
    showToast('JSON copied to clipboard!', 'success');
  } catch {
    showToast('Failed to copy to clipboard', 'error');
  }
}

function downloadJson() {
  if (!analysisResult) return;
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const filename = `analysis_report_${timestamp}.json`;
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

function exportReport() {
  if (!analysisResult) return;
  showToast('Export functionality coming soon', 'info');
}

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
    { level: 'DEBUG', message: 'Checking grammar rules', component: 'GrammarAnalyzer' },
    { level: 'DEBUG', message: 'Checking tone and voice', component: 'ToneAnalyzer' },
    { level: 'INFO', message: 'Generating AI-powered suggestions', component: 'SuggestionEngine' },
    { level: 'INFO', message: 'Creating analysis report', component: 'ReportGenerator' },
    { level: 'INFO', message: 'Analysis completed successfully', component: 'DocumentProcessor' },
  ];
  const randomLog = processingLogs[Math.floor(Math.random() * processingLogs.length)];
  return { ...randomLog, timestamp: new Date().toISOString() };
}

function addLogEntry(logEntry) {
  logs.push(logEntry);
  updateLogsDisplay();
  updateLogsCount();
  if (autoScroll) {
    setTimeout(() => {
      if (elements.logsContent) {
        elements.logsContent.scrollTop = elements.logsContent.scrollHeight;
      }
    }, 10);
  }
}

function updateLogsDisplay() {
  if (!elements.logsContent) return;
  if (logs.length === 0) {
    elements.logsContent.innerHTML =
      '<div class="logs-empty">No logs yet. Logs will appear here when processing starts.</div>';
    return;
  }
  const logsHtml = logs
    .map(
      (log) => `
    <div class="log-entry">
      <span class="log-timestamp">${new Date(log.timestamp).toLocaleTimeString()}</span>
      <span class="log-level log-level-${log.level.toLowerCase()}">${log.level}</span>
      ${log.component ? `<span class="log-component">[${log.component}]</span>` : ''}
      <span class="log-message">${log.message}</span>
    </div>
  `
    )
    .join('');
  elements.logsContent.innerHTML = logsHtml;
}

function updateLogsCount() {
  const count = logs.length;
  elements.logsCount && (elements.logsCount.textContent = `${count} log entries`);
  elements.logsCountBadge && (elements.logsCountBadge.textContent = `${count} entries`);
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
  if (elements.logsStatus) {
    elements.logsStatus.textContent = isLogsRunning ? 'Running' : 'Paused';
    elements.logsStatus.className = isLogsRunning ? 'badge badge-primary' : 'badge badge-secondary';
  }
  elements.pauseIcon?.classList.toggle('hidden', !isLogsRunning);
  elements.playIcon?.classList.toggle('hidden', isLogsRunning);
}

function toggleAutoScroll() {
  autoScroll = !autoScroll;
  elements.autoScrollToggle && (elements.autoScrollToggle.textContent = `Auto-scroll: ${autoScroll ? 'On' : 'Off'}`);
}

function clearLogs() {
  logs = [];
  updateLogsDisplay();
  updateLogsCount();
}

function downloadLogs() {
  if (logs.length === 0) return;
  const logText = logs
    .map(
      (log) => `[${log.timestamp}] ${log.level} ${log.component ? `[${log.component}]` : ''} ${log.message}`
    )
    .join('\n');
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

function showLogsPage() {
  elements.mainPage?.classList.add('hidden');
  elements.logsPage?.classList.remove('hidden');
  if (processingStatus !== 'processing') updateLogsDisplay();
}

function showMainPage() {
  elements.logsPage?.classList.add('hidden');
  elements.mainPage?.classList.remove('hidden');
  if (processingStatus !== 'processing' && logsInterval) {
    clearInterval(logsInterval);
    logsInterval = null;
  }
}

function showToast(message, type = 'info') {
  const container = ensureToastContainer();
  const toast = document.createElement('div');
  toast.className = `toast toast-${type} toast--in`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.remove('toast--in');
    toast.classList.add('toast--out');
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 300);
  }, 3000);
}

function setupEventListeners() {
  if (elements.fileUploadArea) {
    elements.fileUploadArea.addEventListener('click', (e) => {
      if (e.target === e.currentTarget) {
        elements.fileInput?.click();
      }
    });
  }

  elements.browseBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    elements.fileInput?.click();
  });

  elements.fileInput?.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  });

  elements.analyzeBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    startAnalysis();
  });

  elements.clearFileBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
  });

  elements.visualViewBtn?.addEventListener('click', () => toggleView('visual'));
  elements.jsonViewBtn?.addEventListener('click', () => toggleView('json'));
  elements.copyJsonBtn?.addEventListener('click', copyToClipboard);
  elements.downloadJsonBtn?.addEventListener('click', downloadJson);
  elements.exportReportBtn?.addEventListener('click', exportReport);
  elements.severityFilter?.addEventListener('change', filterFindings);
  elements.categoryFilter?.addEventListener('change', filterFindings);
  elements.logsBtn?.addEventListener('click', showLogsPage);
  elements.backBtn?.addEventListener('click', showMainPage);
  elements.themeToggle?.addEventListener('click', toggleTheme);
  elements.themeToggleLogs?.addEventListener('click', toggleTheme);
  elements.toggleLogsBtn?.addEventListener('click', toggleLogs);
  elements.autoScrollToggle?.addEventListener('click', toggleAutoScroll);
  elements.clearLogsBtn?.addEventListener('click', clearLogs);
  elements.downloadLogsBtn?.addEventListener('click', downloadLogs);
}


function init() {
  initTheme();
  setupEventListeners();
  setupDragAndDrop();
  updateStatus('idle');
  updateTimes();
  updateLogsCount();
  testBackendConnection();
  console.log('Document Analysis Application initialized');
}

document.addEventListener('DOMContentLoaded', init);