let currentSession = {
    modelId: null,
    fileInfo: null,
    sessionId: null,
    resultId: null
};

let pollingInterval = null;

// Model Selection
function selectModel(modelId) {
    // Reset previous selection
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // Select new model
    const modelCard = document.querySelector(`.model-card[data-model-id="${modelId}"]`);
    if (modelCard) {
        modelCard.classList.add('selected');
    }
    
    currentSession.modelId = modelId;
    
    // Update selected model info
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            const model = data.models.find(m => m.id === modelId);
            if (model) {
                document.getElementById('selectedModelInfo').style.display = 'block';
                document.getElementById('selectedModelDetails').innerHTML = `
                    <div class="model-details">
                        <h4>${model.name}</h4>
                        <p><i class="fas fa-info-circle"></i> ${model.description}</p>
                        <div class="model-stats">
                            <span class="stat"><i class="fas fa-calculator"></i> Parameters: ${model.parameters}</span>
                            <span class="stat"><i class="fas fa-chart-line"></i> Accuracy: ${model.accuracy}</span>
                            <span class="stat"><i class="fas fa-tachometer-alt"></i> Improvement: ${model.ber_improvement}</span>
                        </div>
                    </div>
                `;
            }
        });
    
    updateRunButtonState();
}

// File Upload
document.getElementById('fileInput').addEventListener('change', handleFileUpload);
document.getElementById('uploadArea').addEventListener('click', () => {
    document.getElementById('fileInput').click();
});
document.getElementById('uploadArea').addEventListener('dragover', (e) => {
    e.preventDefault();
    e.currentTarget.style.borderColor = '#6366f1';
    e.currentTarget.style.background = 'rgba(99, 102, 241, 0.05)';
});
document.getElementById('uploadArea').addEventListener('dragleave', (e) => {
    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    e.currentTarget.style.background = 'transparent';
});
document.getElementById('uploadArea').addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
        uploadFile(file);
    }
});

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (file) {
        uploadFile(file);
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Показываем индикатор загрузки
    updateSystemStatus('Uploading file...');
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData,
        // Не устанавливайте Content-Type - браузер сделает это сам
        // с правильным boundary для multipart/form-data
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            currentSession.fileInfo = data;
            currentSession.sessionId = data.session_id;
            
            // Update file info display
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('fileDetails').innerHTML = `
                <div class="file-details">
                    <p><i class="fas fa-file"></i> <strong>Filename:</strong> ${data.filename}</p>
                    <p><i class="fas fa-hashtag"></i> <strong>Rows:</strong> ${data.stats.rows.toLocaleString()}</p>
                    <p><i class="fas fa-columns"></i> <strong>Columns:</strong> ${data.stats.columns}</p>
                    <p><i class="fas fa-check-circle"></i> <strong>Has TX Data:</strong> ${data.stats.has_tx ? 'Yes' : 'No'}</p>
                    <div class="file-preview">
                        <h5>Preview (first 5 rows):</h5>
                        <pre>${JSON.stringify(data.stats.preview.slice(0, 5), null, 2)}</pre>
                    </div>
                </div>
            `;
            
            updateSystemStatus('File uploaded successfully');
            updateRunButtonState();
        } else {
            alert(`Upload failed: ${data.error}`);
            updateSystemStatus('Upload failed');
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        updateSystemStatus(`Upload failed: ${error.message}`);
    });
}
// Run Inference
function updateRunButtonState() {
    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = !(currentSession.modelId && currentSession.fileInfo);
}

function runInference() {
    if (!currentSession.modelId || !currentSession.fileInfo) {
        alert('Please select a model and upload a file first');
        return;
    }
    
    const batchSize = document.getElementById('batchSize').value;
    const device = document.getElementById('deviceSelect').value;
    
    // Show progress bar
    document.getElementById('progressContainer').style.display = 'block';
    updateProgress(0, 'Starting inference...');
    
    // Disable run button
    document.getElementById('runBtn').disabled = true;
    document.getElementById('downloadBtn').disabled = true;
    
    // Start inference
    fetch('/api/inference', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: currentSession.sessionId,
            model_id: currentSession.modelId,
            filename: currentSession.fileInfo.filename,
            batch_size: parseInt(batchSize),
            device: device
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentSession.resultId = data.result_id;
            updateSystemStatus('Inference started in background');
            
            // Start polling for results
            startPollingResults(data.result_id);
        } else {
            alert(`Inference failed to start: ${data.error}`);
            resetProgress();
        }
    })
    .catch(error => {
        console.error('Inference error:', error);
        updateSystemStatus('Inference failed');
        resetProgress();
    });
}

function startPollingResults(resultId) {
    let elapsed = 0;
    let progress = 0;
    
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    pollingInterval = setInterval(() => {
        elapsed += 1;
        updateProgress(progress, `Processing... (${elapsed}s)`);
        
        // Simulate progress (in real app, this would come from server)
        if (progress < 90) {
            progress += Math.random() * 10;
            progress = Math.min(progress, 90);
        }
        
        // Poll for results
        fetch(`/api/results/${resultId}`)
            .then(response => response.json())
            .then(data => {
                if (data.success && data.result) {
                    if (data.result.success) {
                        // Inference complete
                        clearInterval(pollingInterval);
                        updateProgress(100, 'Complete!');
                        updateSystemStatus('Inference completed successfully');
                        displayResults(data.result);
                        document.getElementById('downloadBtn').disabled = false;
                    } else if (data.result.error) {
                        // Error occurred
                        clearInterval(pollingInterval);
                        updateSystemStatus(`Error: ${data.result.error}`);
                        resetProgress();
                    }
                }
            })
            .catch(error => {
                console.error('Polling error:', error);
            });
    }, 1000);
}

function updateProgress(percent, status) {
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    const progressTime = document.getElementById('progressTime');
    const progressStatus = document.getElementById('progressStatus');
    
    progressFill.style.width = `${percent}%`;
    progressPercent.textContent = `${Math.round(percent)}%`;
    progressTime.textContent = `Elapsed: ${Math.floor(percent / 10)}s`;
    progressStatus.textContent = status;
}

function resetProgress() {
    document.getElementById('progressContainer').style.display = 'none';
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('runBtn').disabled = false;
}

// Display Results
function displayResults(result) {
    // Update metrics
    const metrics = result.metrics;
    if (metrics) {
        document.getElementById('baselineBER').textContent = 
            metrics.baseline_ber ? metrics.baseline_ber.toExponential(2) : 'N/A';
        document.getElementById('equalizedBER').textContent = 
            metrics.equalized_ber ? metrics.equalized_ber.toExponential(2) : 'N/A';
        document.getElementById('improvement').textContent = 
            metrics.improvement_rel ? `${metrics.improvement_rel.toFixed(1)}%` : 'N/A';
        document.getElementById('snrGain').textContent = 
            metrics.improvement_db ? `${metrics.improvement_db.toFixed(2)} dB` : 'N/A';
    }
    
    // Update detailed info
    document.getElementById('symbolsCount').textContent = 
        result.predictions.num_predictions.toLocaleString();
    document.getElementById('processingTime').textContent = 
        `${result.processing_time.toFixed(2)}s`;
    document.getElementById('modelType').textContent = 
        result.model_info.name;
    document.getElementById('dataStatus').textContent = 
        result.data.has_tx ? 'With TX data' : 'RX data only';
    
    // Update raw data preview
    const previewSymbols = result.predictions.symbols.slice(0, 10);
    const previewBits = result.predictions.bits.slice(0, 10);
    document.getElementById('rawDataPreview').textContent = 
        JSON.stringify({symbols: previewSymbols, bits: previewBits}, null, 2);
    
    // Display plots
    if (result.plots) {
        displayPlot('constellation', result.plots.constellation);
    }
    
    // Enable download button
    document.getElementById('downloadBtn').disabled = false;
}

function displayPlot(plotType, plotData) {
    if (!plotData) return;
    
    const plotContainer = document.getElementById('plotContainer');
    Plotly.newPlot(plotContainer, plotData.data, plotData.layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: `plot_${plotType}`,
            height: 600,
            width: 800,
            scale: 2
        }
    });
}

function switchPlot(plotType) {
    // Update active tab
    document.querySelectorAll('.plot-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.currentTarget.classList.add('active');
    
    // Fetch and display plot
    if (currentSession.resultId) {
        fetch(`/api/results/${currentSession.resultId}`)
            .then(response => response.json())
            .then(data => {
                if (data.success && data.result && data.result.plots) {
                    switch(plotType) {
                        case 'constellation':
                            displayPlot('constellation', data.result.plots.constellation);
                            break;
                        case 'ber':
                            displayPlot('ber', data.result.plots.ber_comparison);
                            break;
                        case 'time':
                            displayPlot('time', data.result.plots.time_series);
                            break;
                    }
                }
            });
    }
}

// Download Results
function downloadResults() {
    if (!currentSession.resultId) return;
    
    window.location.href = `/api/download/${currentSession.resultId}`;
}

// Refresh Results
function refreshResults() {
    if (!currentSession.resultId) return;
    
    fetch(`/api/results/${currentSession.resultId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.result) {
                displayResults(data.result);
                updateSystemStatus('Results refreshed');
            }
        });
}

// System Status
function updateSystemStatus(message) {
    document.getElementById('systemStatus').textContent = message;
    
    // Add notification effect
    const statusElement = document.getElementById('systemStatus');
    statusElement.classList.remove('fade-in');
    void statusElement.offsetWidth; // Trigger reflow
    statusElement.classList.add('fade-in');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Initialize range input display
    const batchSizeInput = document.getElementById('batchSize');
    const batchSizeValue = document.getElementById('batchSizeValue');
    
    batchSizeInput.addEventListener('input', () => {
        batchSizeValue.textContent = batchSizeInput.value;
    });
    
    // Check system status
    updateSystemStatus('Ready to start');
    
    // Load models
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('Models loaded:', data.models.length);
            }
        });
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl + Space to run inference
    if (e.ctrlKey && e.code === 'Space') {
        e.preventDefault();
        runInference();
    }
    
    // Ctrl + D to download
    if (e.ctrlKey && e.code === 'KeyD') {
        e.preventDefault();
        downloadResults();
    }
    
    // Ctrl + R to refresh
    if (e.ctrlKey && e.code === 'KeyR') {
        e.preventDefault();
        refreshResults();
    }
});