const API_BASE_URL = window.location.origin;

// Authentication state
let currentUser = null;

// Application state
let currentFile = null;
let analysisResult = null;
let processingStatus = 'idle';
let startTime = null;
let endTime = null;
let logs = [];
let isLogsRunning = true;
let autoScroll = true;
let es = null;
let progressEs = null;
let currentView = 'visual';
let filteredFindings = [];
let currentFileId = null;

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
  metaUploader: document.getElementById('meta-uploader'),
};

function updateAnalyzeEnabled() {
  const hasFile = !!elements.fileInput?.files?.[0];
  const hasName = !!elements.userInfo?.value?.trim();
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
  updateAnalyzeEnabled();
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

function connectProgressStream(fileId) {
  if (progressEs) {
    try { progressEs.close(); } catch (_) {}
    progressEs = null;
  }

  progressEs = new EventSource(`${API_BASE_URL}/progress/stream/${fileId}`);

  progressEs.onopen = () => {
    console.log('Progress stream connected');
  };

  progressEs.onerror = () => {
    console.log('Progress stream error');
  };

  progressEs.onmessage = (evt) => {
    if (!evt || !evt.data) return;
    try {
      const progress = JSON.parse(evt.data);
      console.log('Progress update:', progress);

      // Update status message based on progress stage
      updateStatus('processing', progress.message);

      // Close stream when completed
      if (progress.stage === 'completed') {
        setTimeout(() => {
          if (progressEs) {
            progressEs.close();
            progressEs = null;
          }
        }, 1000);
      }
    } catch (e) {
      console.error('Failed to parse progress:', e);
    }
  };
}

function closeProgressStream() {
  if (progressEs) {
    try { progressEs.close(); } catch (_) {}
    progressEs = null;
  }
}

async function waitForProcessingComplete(fileId) {
  /**
   * Poll the backend for processing completion.
   * The progress stream provides real-time updates,
   * but we also poll to get the final result.
   */
  const maxAttempts = 300; // 5 minutes with 1s intervals
  let attempts = 0;

  console.log('Starting to poll for completion, fileId:', fileId);

  while (attempts < maxAttempts) {
    try {
      const statusResponse = await fetch(`${API_BASE_URL}/processing-status/${fileId}`);
      console.log(`Poll attempt ${attempts + 1}, status code:`, statusResponse.status);

      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        console.log('Status data:', statusData);

        if (statusData.status === 'completed' && statusData.result) {
          // Processing complete!
          console.log('Processing completed! Result received:', statusData.result);
          analysisResult = statusData.result;
          console.log('analysisResult set to:', analysisResult);

          endTime = new Date();
          updateStatus('completed', 'Document analysis completed successfully');
          updateTimes();

          console.log('About to call displayVisualReport()');
          displayVisualReport();
          console.log('displayVisualReport() completed');

          console.log('About to call displayJsonResult()');
          displayJsonResult();
          console.log('displayJsonResult() completed');

          elements.visualReport?.classList.remove('hidden');
          console.log('Visual report shown');

          elements.analyzeBtn.disabled = false;
          elements.analyzeBtn.textContent = 'Analyze Document';
          showToast('Document processed successfully!', 'success');
          closeProgressStream();
          return;
        } else if (statusData.status === 'error') {
          console.error('Processing error:', statusData.error);
          throw new Error(statusData.error || 'Processing failed');
        }
        // Still processing, continue polling
        console.log('Still processing, will poll again...');
      } else {
        console.warn('Status check returned non-OK:', statusResponse.status);
      }
    } catch (error) {
      console.error('Error checking processing status:', error);
      // Continue polling unless it's a fatal error
      if (attempts > 10) {
        throw error;
      }
    }

    // Wait 1 second before next poll
    await new Promise(resolve => setTimeout(resolve, 1000));
    attempts++;
  }

  console.error('Processing timeout after', attempts, 'attempts');
  throw new Error('Processing timeout - took longer than expected');
}

async function startAnalysis() {
  if (!currentFile) return;
  console.log('Starting analysis for:', currentFile.name, 'User:', elements.userInfo?.value);

  // Show the status line when analysis starts
  elements.statusLine?.classList.remove('hidden');

  // Immediately reflect "Uploaded by" in the UI (pre-backend)
  elements.metaUploader.textContent = elements.userInfo.value.trim() || '--';

  startTime = new Date();
  endTime = null;
  updateStatus('processing', 'Preparing to upload...');
  updateTimes();
  elements.analyzeBtn.disabled = true;
  elements.analyzeBtn.textContent = 'Processing...';
  elements.visualReport?.classList.add('hidden');
  connectLogsStream();

  try {
    const healthResponse = await fetch(`${API_BASE_URL}/health`);
    if (!healthResponse.ok) throw new Error('Backend not available');

    // Get next file_id before upload to enable progress tracking
    const fileIdResponse = await fetch(`${API_BASE_URL}/next-document-id`);
    if (!fileIdResponse.ok) throw new Error('Failed to get document ID');

    const {file_id} = await fileIdResponse.json();
    currentFileId = file_id;

    // Connect to progress stream BEFORE starting upload
    connectProgressStream(file_id);

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('user_info', elements.userInfo?.value);

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

    const uploadResult = await uploadResponse.json();

    // Check if processing started (202) or completed immediately (cached)
    if (uploadResult.status === 'processing') {
      // Processing started - progress stream will update status
      // Poll for completion
      await waitForProcessingComplete(uploadResult.file_id);
    } else if (uploadResult.status === 'completed') {
      // Cached result returned immediately
      analysisResult = uploadResult.result;
      endTime = new Date();
      updateStatus('completed', 'Document analysis completed successfully');
      updateTimes();
      displayVisualReport();
      displayJsonResult();
      elements.visualReport?.classList.remove('hidden');
      elements.analyzeBtn.disabled = false;
      elements.analyzeBtn.textContent = 'Analyze Document';
      showToast('Document processed successfully!', 'success');
    }
  } catch (error) {
    console.error('Analysis error:', error);
    endTime = new Date();
    updateStatus('error', `Analysis failed: ${error.message}`);
    updateTimes();
    showToast(`Analysis failed: ${error.message}`, 'error');
    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.textContent = 'Analyze Document';
    closeProgressStream();
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
  const metaType     = document.getElementById('meta-type');
  const metaAuthor   = document.getElementById('meta-author');
  const metaSlides   = document.getElementById('meta-slides');
  const metaTimestamp= document.getElementById('meta-timestamp');

  const uploadedBy =
  analysisResult?.metadata?.uploaded_by ||
  analysisResult?.user_info ||
  elements.userInfo?.value?.trim() ||
  '--';

  if (metaFilename) metaFilename.textContent = analysisResult.original_filename || '--';
  if (metaType)     metaType.textContent     = metadata.document_type?.toUpperCase() || '--';
  if (metaAuthor)   metaAuthor.textContent   = metadata.author || '--';
  if (metaSlides)   metaSlides.textContent   = analysisResult.content_statistics?.total_pages || '--';
  if (metaTimestamp)metaTimestamp.textContent= new Date(analysisResult.processed_at || Date.now()).toLocaleString();
  if (elements.metaUploader) elements.metaUploader.textContent = uploadedBy;


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

    const summaryText = finding.found_text || finding.description || 'No text available';
    const truncatedText = summaryText.length > 80 ? summaryText.substring(0, 80) + '...' : summaryText;

    const severity = finding.severity || 'info';

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
          <div class="finding-suggestion">${(finding.suggestion || 'No suggestion available').replace(/\n/g, '<br>')}</div>
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

function connectLogsStream() {
  if (es) { try { es.close(); } catch (_) {} es = null; }

  if (elements.logsStatus) {
    elements.logsStatus.textContent = 'Connecting…';
    elements.logsStatus.className = 'badge badge-secondary';
  }

  es = new EventSource(`${API_BASE_URL}/logs/stream`);

  es.onopen = () => {
    if (elements.logsStatus) {
      elements.logsStatus.textContent = 'Running';
      elements.logsStatus.className = 'badge badge-primary';
    }
  };

  es.onerror = () => {
    if (elements.logsStatus) {
      elements.logsStatus.textContent = 'Reconnecting…';
      elements.logsStatus.className = 'badge badge-secondary';
    }
  };

  es.onmessage = (evt) => {
    if (!evt || !evt.data) return;
    try {
      const entry = JSON.parse(evt.data);
      if (isLogsRunning) addLogEntry(entry);
    } catch (_) {}
  };
}

function closeLogsStream() {
  if (es) { try { es.close(); } catch (_) {} es = null; }
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
  connectLogsStream();
  if (processingStatus !== 'processing') updateLogsDisplay();
}

function showMainPage() {
  elements.logsPage?.classList.add('hidden');
  elements.mainPage?.classList.remove('hidden');
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
    updateAnalyzeEnabled(); // keep button disabled until name/email present
  });

  elements.analyzeBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!elements.userInfo?.value?.trim()) {
      elements.userInfo?.reportValidity?.(); // shows native tooltip if `required` is set
      return;
    }
    if (!elements.fileInput?.files?.[0]) {
      showToast('Please choose a .pptx or .pdf file', 'error');
      return;
    }
    if (elements.metaUploader) {
      elements.metaUploader.textContent = elements.userInfo.value.trim() || '--';
    }
    startAnalysis();
  });


  elements.clearFileBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
    updateAnalyzeEnabled();
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


// Authentication check
async function checkAuthentication() {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      method: 'GET',
      credentials: 'include'
    });

    if (!response.ok) {
      console.warn('Not authenticated, redirecting to login...');
      window.location.href = '/pages/login.html';
      return false;
    }

    const data = await response.json();

    if (!data.authenticated) {
      console.warn('User not authenticated, redirecting to login...');
      window.location.href = '/pages/login.html';
      return false;
    }

    // User is authenticated
    currentUser = data.user;
    console.log('User authenticated:', currentUser.email);
    updateUserDisplay();
    return true;

  } catch (error) {
    console.error('Authentication check failed:', error);
    window.location.href = '/pages/login.html';
    return false;
  }
}

// Update user display in UI
function updateUserDisplay() {
  if (!currentUser) return;

  // Update meta-uploader field if it exists
  if (elements.metaUploader) {
    elements.metaUploader.value = currentUser.display_name || currentUser.email;
  }

  // Update user info in header
  const userInfoDisplay = document.getElementById('user-info-display');
  if (userInfoDisplay) {
    userInfoDisplay.innerHTML = `
      <div class="user-info-text">
        <div class="user-name">${currentUser.display_name || 'User'}</div>
        <div class="user-email">${currentUser.email}</div>
      </div>
    `;
  }

  // Add logout event listener to header button
  const logoutBtnHeader = document.getElementById('logout-btn-header');
  if (logoutBtnHeader && !logoutBtnHeader.hasAttribute('data-listener')) {
    logoutBtnHeader.addEventListener('click', handleLogout);
    logoutBtnHeader.setAttribute('data-listener', 'true');
  }

  // Show admin buttons if user is admin
  if (currentUser.role === 'ADMIN') {
    const adminDashboardBtn = document.getElementById('admin-dashboard-btn');
    const logsBtn = document.getElementById('logs-btn');

    if (adminDashboardBtn) {
      adminDashboardBtn.style.display = 'flex';
      adminDashboardBtn.classList.remove('hidden');
      adminDashboardBtn.addEventListener('click', () => {
        window.location.href = '/pages/admin.html';
      });
    }

    if (logsBtn) {
      logsBtn.style.display = 'flex';
      logsBtn.classList.remove('hidden');
    }
  }
}

// Handle logout
async function handleLogout() {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/logout`, {
      method: 'POST',
      credentials: 'include'
    });

    if (response.ok) {
      console.log('Logged out successfully');
      window.location.href = '/pages/login.html';
    } else {
      console.error('Logout failed');
      // Redirect anyway
      window.location.href = '/pages/login.html';
    }
  } catch (error) {
    console.error('Logout error:', error);
    // Redirect anyway
    window.location.href = '/pages/login.html';
  }
}

async function init() {
  // Check authentication first
  const isAuthenticated = await checkAuthentication();
  if (!isAuthenticated) {
    return; // Stop initialization if not authenticated
  }

  // Continue with normal initialization
  initTheme();
  setupEventListeners();
  setupDragAndDrop();
  updateLogsCount();
  testBackendConnection();
  console.log('Document Analysis Application initialized');
}

document.addEventListener('DOMContentLoaded', init);
window.addEventListener('beforeunload', () => {
  try { closeLogsStream(); } catch {}
  try { closeProgressStream(); } catch {}
});
