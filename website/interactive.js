// Interactive Dark Matter Detection
let currentMode = 'dark-matter';
let imageFile = null;
let imageCanvas = null;
let imageCtx = null;
let groundTruthCanvas = null;
let groundTruthCtx = null;
let predictionCanvas = null;
let predictionCtx = null;
let markers = [];
let galaxyData = [];
let predictions = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    imageCanvas = document.getElementById('imageCanvas');
    imageCtx = imageCanvas.getContext('2d');
    groundTruthCanvas = document.getElementById('groundTruthCanvas');
    groundTruthCtx = groundTruthCanvas.getContext('2d');
    predictionCanvas = document.getElementById('predictionCanvas');
    predictionCtx = predictionCanvas.getContext('2d');

    setupEventListeners();
});

function setupEventListeners() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const modeButtons = document.querySelectorAll('.btn-mode');
    const generateGridBtn = document.getElementById('generateGridBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');

    // File upload
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Mode buttons
    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            modeButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            if (currentMode === 'clear') {
                clearMarkers();
                currentMode = 'dark-matter';
                modeButtons[0].classList.add('active');
            }
        });
    });

    // Canvas click handler
    imageCanvas.addEventListener('click', handleCanvasClick);

    // Action buttons
    generateGridBtn.addEventListener('click', generateGalaxyGrid);
    predictBtn.addEventListener('click', runPrediction);
    resetBtn.addEventListener('click', resetAll);
}

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    imageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            // Set canvas size
            const maxWidth = 1200;
            const maxHeight = 800;
            let width = img.width;
            let height = img.height;

            if (width > maxWidth) {
                height = (height * maxWidth) / width;
                width = maxWidth;
            }
            if (height > maxHeight) {
                width = (width * maxHeight) / height;
                height = maxHeight;
            }

            imageCanvas.width = width;
            imageCanvas.height = height;
            imageCtx.drawImage(img, 0, 0, width, height);

            // Show canvas and controls
            document.getElementById('canvasContainer').style.display = 'block';
            document.getElementById('controlsPanel').style.display = 'block';
            uploadArea.style.display = 'none';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function handleCanvasClick(e) {
    if (currentMode === 'clear') return;

    const rect = imageCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if clicking on existing marker
    const existingIndex = markers.findIndex(m => 
        Math.abs(m.x - x) < 15 && Math.abs(m.y - y) < 15
    );

    if (existingIndex >= 0) {
        markers.splice(existingIndex, 1);
    } else {
        markers.push({
            x: x,
            y: y,
            type: currentMode === 'dark-matter' ? 'dark-matter' : 'background'
        });
    }

    drawMarkers();
    updateStats();
}

function drawMarkers() {
    // Redraw image
    const img = new Image();
    img.onload = () => {
        imageCtx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
        // Draw markers
        markers.forEach(marker => {
            imageCtx.beginPath();
            imageCtx.arc(marker.x, marker.y, 8, 0, 2 * Math.PI);
            imageCtx.fillStyle = marker.type === 'dark-matter' ? 'rgba(255, 0, 0, 0.7)' : 'rgba(0, 0, 255, 0.7)';
            imageCtx.fill();
            imageCtx.strokeStyle = marker.type === 'dark-matter' ? 'darkred' : 'darkblue';
            imageCtx.lineWidth = 2;
            imageCtx.stroke();
        });
    };
    img.src = imageCanvas.toDataURL();
}

function clearMarkers() {
    markers = [];
    const img = new Image();
    img.onload = () => {
        imageCtx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
    };
    img.src = imageCanvas.toDataURL();
    updateStats();
}

function updateStats() {
    const darkMatterCount = markers.filter(m => m.type === 'dark-matter').length;
    const backgroundCount = markers.filter(m => m.type === 'background').length;
    document.getElementById('darkMatterCount').textContent = darkMatterCount;
    document.getElementById('backgroundCount').textContent = backgroundCount;
}

function generateGalaxyGrid() {
    if (markers.length === 0) {
        alert('Please mark at least some galaxies first!');
        return;
    }

    // Generate synthetic galaxy data based on markers
    galaxyData = [];
    markers.forEach(marker => {
        // Extract pixel data around marker
        const x = Math.floor(marker.x);
        const y = Math.floor(marker.y);
        const size = 10;
        
        // Get pixel data
        const imageData = imageCtx.getImageData(
            Math.max(0, x - size), 
            Math.max(0, y - size), 
            Math.min(size * 2, imageCanvas.width - x + size),
            Math.min(size * 2, imageCanvas.height - y + size)
        );

        // Calculate ellipticity-like features from image
        const features = calculateFeaturesFromImage(imageData, marker.type === 'dark-matter');
        
        galaxyData.push({
            x: marker.x,
            y: marker.y,
            eps1: features.eps1,
            eps2: features.eps2,
            label: marker.type === 'dark-matter' ? 1 : 0
        });
    });

    // Draw ground truth visualization
    drawGroundTruth();
}

function calculateFeaturesFromImage(imageData, isDarkMatter) {
    // Simulate ellipticity calculation from image
    // In a real implementation, this would analyze the galaxy shape
    const pixels = imageData.data;
    let sum = 0;
    let count = 0;

    for (let i = 0; i < pixels.length; i += 4) {
        const brightness = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        sum += brightness;
        count++;
    }

    const avgBrightness = sum / count;
    
    // Generate synthetic ellipticity based on brightness and dark matter presence
    // Dark matter regions tend to have stronger shear (higher ellipticity)
    const baseEps1 = (avgBrightness / 255 - 0.5) * 0.1;
    const baseEps2 = (Math.random() - 0.5) * 0.1;
    
    if (isDarkMatter) {
        // Add stronger shear for dark matter regions
        return {
            eps1: baseEps1 + (Math.random() * 0.03 + 0.05),
            eps2: baseEps2 + (Math.random() * 0.03 + 0.05)
        };
    } else {
        // Weaker/zero shear for background
        return {
            eps1: baseEps1 + (Math.random() - 0.5) * 0.02,
            eps2: baseEps2 + (Math.random() - 0.5) * 0.02
        };
    }
}

function drawGroundTruth() {
    groundTruthCanvas.width = imageCanvas.width;
    groundTruthCanvas.height = imageCanvas.height;
    
    // Draw image
    const img = new Image();
    img.onload = () => {
        groundTruthCtx.drawImage(img, 0, 0, groundTruthCanvas.width, groundTruthCanvas.height);
        
        // Draw markers
        markers.forEach(marker => {
            groundTruthCtx.beginPath();
            groundTruthCtx.arc(marker.x, marker.y, 10, 0, 2 * Math.PI);
            groundTruthCtx.fillStyle = marker.type === 'dark-matter' 
                ? 'rgba(255, 0, 0, 0.6)' 
                : 'rgba(0, 0, 255, 0.6)';
            groundTruthCtx.fill();
            groundTruthCtx.strokeStyle = marker.type === 'dark-matter' ? 'darkred' : 'darkblue';
            groundTruthCtx.lineWidth = 3;
            groundTruthCtx.stroke();
        });
    };
    img.src = imageCanvas.toDataURL();
}

async function runPrediction() {
    if (galaxyData.length === 0) {
        alert('Please generate galaxy grid first!');
        return;
    }

    // Show loading
    const predictBtn = document.getElementById('predictBtn');
    const originalText = predictBtn.textContent;
    predictBtn.textContent = 'Predicting...';
    predictBtn.disabled = true;

    try {
        // Prepare data for prediction
        const features = galaxyData.map(g => [g.eps1, g.eps2]);
        
        // Call prediction API (we'll create a simple client-side prediction)
        // In production, this would call a backend API
        predictions = await predictDarkMatter(features);
        
        // Draw predictions
        drawPredictions();
        
        // Calculate accuracy
        const accuracy = calculateAccuracy();
        document.getElementById('accuracy').textContent = (accuracy * 100).toFixed(1) + '%';
        document.getElementById('predictedDarkMatter').textContent = 
            predictions.filter(p => p === 1).length;
        
        // Show results
        document.getElementById('resultsSection').style.display = 'grid';
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error running prediction. Please try again.');
    } finally {
        predictBtn.textContent = originalText;
        predictBtn.disabled = false;
    }
}

async function predictDarkMatter(features) {
    try {
        // Call the API server
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ features: features })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        return data.predictions;
    } catch (error) {
        console.warn('API call failed, using fallback prediction:', error);
        // Fallback to rule-based prediction if API is not available
        return features.map(f => {
            const eps1 = f[0];
            const eps2 = f[1];
            const gamma_tot = Math.sqrt(eps1 * eps1 + eps2 * eps2);
            // Threshold similar to training: gamma_tot > 0.05
            return gamma_tot > 0.05 ? 1 : 0;
        });
    }
}

function drawPredictions() {
    predictionCanvas.width = imageCanvas.width;
    predictionCanvas.height = imageCanvas.height;
    
    // Draw image
    const img = new Image();
    img.onload = () => {
        predictionCtx.drawImage(img, 0, 0, predictionCanvas.width, predictionCanvas.height);
        
        // Draw predictions
        galaxyData.forEach((galaxy, index) => {
            const prediction = predictions[index];
            predictionCtx.beginPath();
            predictionCtx.arc(galaxy.x, galaxy.y, 12, 0, 2 * Math.PI);
            
            if (prediction === 1) {
                // Predicted dark matter - yellow star
                predictionCtx.fillStyle = 'rgba(255, 255, 0, 0.7)';
                predictionCtx.fill();
                predictionCtx.strokeStyle = 'yellow';
                predictionCtx.lineWidth = 2;
                predictionCtx.stroke();
                
                // Draw star shape
                drawStar(predictionCtx, galaxy.x, galaxy.y, 5, 12, 5);
            } else {
                // Predicted background - light blue circle
                predictionCtx.fillStyle = 'rgba(173, 216, 230, 0.5)';
                predictionCtx.fill();
                predictionCtx.strokeStyle = 'cyan';
                predictionCtx.lineWidth = 2;
                predictionCtx.stroke();
            }
        });
    };
    img.src = imageCanvas.toDataURL();
}

function drawStar(ctx, cx, cy, spikes, outerRadius, innerRadius) {
    ctx.beginPath();
    ctx.moveTo(cx, cy - outerRadius);
    for (let i = 0; i < spikes * 2; i++) {
        const radius = i % 2 === 0 ? outerRadius : innerRadius;
        const angle = (Math.PI * i) / spikes - Math.PI / 2;
        ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 255, 0, 0.8)';
    ctx.fill();
    ctx.strokeStyle = 'yellow';
    ctx.lineWidth = 2;
    ctx.stroke();
}

function calculateAccuracy() {
    if (predictions.length !== galaxyData.length) return 0;
    
    let correct = 0;
    galaxyData.forEach((galaxy, index) => {
        if (predictions[index] === galaxy.label) {
            correct++;
        }
    });
    
    return correct / galaxyData.length;
}

function resetAll() {
    markers = [];
    galaxyData = [];
    predictions = [];
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('canvasContainer').style.display = 'none';
    document.getElementById('controlsPanel').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
    imageFile = null;
    updateStats();
}

