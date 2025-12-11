// Interactive Dark Matter Detection
let currentMode = 'dark-matter';
let imageFile = null;
let imageCanvas = null;
let imageCtx = null;
let groundTruthCanvas = null;
let groundTruthCtx = null;
let predictionCanvas = null;
let predictionCtx = null;
let lensingCanvas = null;
let lensingCtx = null;
let markers = []; // Only dark matter markers now
let galaxyData = [];
let predictions = [];
let originalImageData = null; // Store original image for lensing effect

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    imageCanvas = document.getElementById('imageCanvas');
    imageCtx = imageCanvas.getContext('2d');
    groundTruthCanvas = document.getElementById('groundTruthCanvas');
    groundTruthCtx = groundTruthCanvas.getContext('2d');
    predictionCanvas = document.getElementById('predictionCanvas');
    predictionCtx = predictionCanvas.getContext('2d');
    lensingCanvas = document.getElementById('lensingCanvas');
    lensingCtx = lensingCanvas.getContext('2d');

    setupEventListeners();
});

// Example images - using canvas-generated galaxy fields
const exampleImages = {
    example1: null, // Will be generated
    example2: null,
    example3: null
};

function setupEventListeners() {
    const modeButtons = document.querySelectorAll('.btn-mode');
    const showLensingBtn = document.getElementById('showLensingBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');
    const exampleCards = document.querySelectorAll('.example-image-card');

    // Example image selection
    exampleCards.forEach(card => {
        card.addEventListener('click', () => {
            const imageType = card.dataset.image;
            exampleCards.forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            loadExampleImage(imageType);
        });
    });

    // Mode buttons (only dark matter marking now)
    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.dataset.mode === 'clear') {
                clearMarkers();
            } else {
                modeButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentMode = btn.dataset.mode;
            }
        });
    });

    // Canvas click handler
    imageCanvas.addEventListener('click', handleCanvasClick);

    // Action buttons
    if (showLensingBtn) {
        showLensingBtn.addEventListener('click', showGravitationalLensing);
    }
    if (predictBtn) {
        predictBtn.addEventListener('click', runPrediction);
    }
    if (resetBtn) {
        resetBtn.addEventListener('click', resetAll);
    }
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
        // Remove marker if clicking on existing one
        markers.splice(existingIndex, 1);
    } else {
        // Add dark matter marker
        markers.push({
            x: x,
            y: y,
            type: 'dark-matter'
        });
    }

    drawMarkers();
    updateStats();
    
    // Show "Show Lensing Effect" button when markers are added
    if (markers.length > 0) {
        document.getElementById('showLensingBtn').style.display = 'inline-block';
    } else {
        document.getElementById('showLensingBtn').style.display = 'none';
    }
}

function drawMarkers() {
    // Redraw original image
    if (originalImageData) {
        imageCtx.putImageData(originalImageData, 0, 0);
    } else {
        const img = new Image();
        img.onload = () => {
            imageCtx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
            // Store original for lensing effect
            originalImageData = imageCtx.getImageData(0, 0, imageCanvas.width, imageCanvas.height);
            drawMarkersOnImage();
        };
        img.src = imageCanvas.toDataURL();
        return;
    }
    drawMarkersOnImage();
}

function drawMarkersOnImage() {
    // Draw red markers for dark matter locations
    markers.forEach(marker => {
        // Draw marker circle
        imageCtx.beginPath();
        imageCtx.arc(marker.x, marker.y, 12, 0, 2 * Math.PI);
        imageCtx.fillStyle = 'rgba(255, 0, 0, 0.6)';
        imageCtx.fill();
        imageCtx.strokeStyle = 'darkred';
        imageCtx.lineWidth = 3;
        imageCtx.stroke();
        
        // Draw pulsing effect
        imageCtx.beginPath();
        imageCtx.arc(marker.x, marker.y, 18, 0, 2 * Math.PI);
        imageCtx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
        imageCtx.lineWidth = 2;
        imageCtx.stroke();
    });
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
    const darkMatterCount = markers.length;
    document.getElementById('darkMatterCount').textContent = darkMatterCount;
}

function showGravitationalLensing() {
    if (markers.length === 0) {
        alert('Please mark at least one dark matter location first!');
        return;
    }

    // Show lensing effect visualization
    lensingCanvas.width = imageCanvas.width;
    lensingCanvas.height = imageCanvas.height;
    
    // Draw original image
    const img = new Image();
    img.onload = () => {
        lensingCtx.drawImage(img, 0, 0, lensingCanvas.width, lensingCanvas.height);
        
        // Apply gravitational lensing distortion effect to galaxies near dark matter markers
        applyLensingEffect();
        
        // Draw markers
        markers.forEach(marker => {
            lensingCtx.beginPath();
            lensingCtx.arc(marker.x, marker.y, 15, 0, 2 * Math.PI);
            lensingCtx.fillStyle = 'rgba(255, 0, 0, 0.5)';
            lensingCtx.fill();
            lensingCtx.strokeStyle = 'darkred';
            lensingCtx.lineWidth = 3;
            lensingCtx.stroke();
            
            // Label
            lensingCtx.fillStyle = 'white';
            lensingCtx.font = 'bold 14px Arial';
            lensingCtx.fillText('DM', marker.x - 8, marker.y - 20);
        });
        
        // Show the lensing section
        document.getElementById('lensingSection').style.display = 'grid';
        document.getElementById('showLensingBtn').style.display = 'none';
        document.getElementById('predictBtn').style.display = 'inline-block';
    };
    img.src = imageCanvas.toDataURL();
}

function applyLensingEffect() {
    // Apply visual distortion to galaxies near dark matter markers
    // This simulates gravitational lensing
    
    const imageData = lensingCtx.getImageData(0, 0, lensingCanvas.width, lensingCanvas.height);
    const data = imageData.data;
    const width = lensingCanvas.width;
    const height = lensingCanvas.height;
    
    // Create new image data for distorted version
    const newData = new ImageData(width, height);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let newX = x;
            let newY = y;
            
            // Check distance to each dark matter marker
            markers.forEach(marker => {
                const dx = x - marker.x;
                const dy = y - marker.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                // Apply lensing distortion within radius
                if (dist < 80 && dist > 0) {
                    const strength = (80 - dist) / 80; // Stronger closer to marker
                    const angle = Math.atan2(dy, dx);
                    
                    // Stretch along radial direction (gravitational lensing effect)
                    const stretch = 1 + strength * 0.3;
                    newX += Math.cos(angle) * strength * 5;
                    newY += Math.sin(angle) * strength * 5;
                    
                    // Also add tangential stretch (makes elliptical)
                    const perpAngle = angle + Math.PI / 2;
                    newX += Math.cos(perpAngle) * strength * 3;
                    newY += Math.sin(perpAngle) * strength * 3;
                }
            });
            
            // Clamp coordinates
            newX = Math.max(0, Math.min(width - 1, Math.round(newX)));
            newY = Math.max(0, Math.min(height - 1, Math.round(newY)));
            
            // Copy pixel
            const origIdx = (y * width + x) * 4;
            const newIdx = (newY * width + newX) * 4;
            
            newData.data[origIdx] = data[newIdx];
            newData.data[origIdx + 1] = data[newIdx + 1];
            newData.data[origIdx + 2] = data[newIdx + 2];
            newData.data[origIdx + 3] = data[newIdx + 3];
        }
    }
    
    // Apply the distorted image
    lensingCtx.putImageData(newData, 0, 0);
    
    // Generate galaxy data from the distorted image
    generateGalaxyDataFromLensing();
}

function generateGalaxyDataFromLensing() {
    // Extract features from the lensed (distorted) image
    galaxyData = [];
    
    // Sample galaxies across the image (not just at markers)
    const samplePoints = [];
    
    // Add points at marker locations
    markers.forEach(marker => {
        samplePoints.push({x: marker.x, y: marker.y, isDarkMatter: true});
    });
    
    // Add random sample points across the image
    for (let i = 0; i < 50; i++) {
        samplePoints.push({
            x: Math.random() * imageCanvas.width,
            y: Math.random() * imageCanvas.height,
            isDarkMatter: false
        });
    }
    
    samplePoints.forEach(point => {
        const x = Math.floor(point.x);
        const y = Math.floor(point.y);
        const size = 15;
        
        // Get pixel data from lensed image
        const imageData = lensingCtx.getImageData(
            Math.max(0, x - size), 
            Math.max(0, y - size), 
            Math.min(size * 2, lensingCanvas.width - x + size),
            Math.min(size * 2, lensingCanvas.height - y + size)
        );

        // Calculate ellipticity features from distorted image
        const features = calculateFeaturesFromLensedImage(imageData, point.isDarkMatter);
        
        galaxyData.push({
            x: point.x,
            y: point.y,
            eps1: features.eps1,
            eps2: features.eps2,
            label: point.isDarkMatter ? 1 : 0
        });
    });
}

function calculateFeaturesFromLensedImage(imageData, isDarkMatter) {
    // Calculate ellipticity from the lensed (distorted) image
    const pixels = imageData.data;
    const width = Math.sqrt(pixels.length / 4);
    const height = width;
    
    // Calculate centroid
    let sumX = 0, sumY = 0, totalBrightness = 0;
    for (let i = 0; i < pixels.length; i += 4) {
        const brightness = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        const x = (i / 4) % width;
        const y = Math.floor((i / 4) / width);
        sumX += x * brightness;
        sumY += y * brightness;
        totalBrightness += brightness;
    }
    
    if (totalBrightness === 0) {
        return {eps1: 0, eps2: 0};
    }
    
    const cx = sumX / totalBrightness;
    const cy = sumY / totalBrightness;
    
    // Calculate shape moments (ellipticity)
    let q11 = 0, q22 = 0, q12 = 0;
    for (let i = 0; i < pixels.length; i += 4) {
        const brightness = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        const x = (i / 4) % width;
        const y = Math.floor((i / 4) / width);
        const dx = x - cx;
        const dy = y - cy;
        const r2 = dx * dx + dy * dy;
        if (r2 > 0) {
            q11 += brightness * dx * dx / r2;
            q22 += brightness * dy * dy / r2;
            q12 += brightness * dx * dy / r2;
        }
    }
    
    // Normalize
    const norm = q11 + q22;
    if (norm === 0) {
        return {eps1: 0, eps2: 0};
    }
    
    // Calculate ellipticity components
    const eps1 = (q11 - q22) / norm;
    const eps2 = (2 * q12) / norm;
    
    // Scale to match training data range
    return {
        eps1: Math.max(-0.1, Math.min(0.1, eps1 * 0.1)),
        eps2: Math.max(-0.1, Math.min(0.1, eps2 * 0.1))
    };
}

function drawGroundTruth() {
    groundTruthCanvas.width = lensingCanvas.width;
    groundTruthCanvas.height = lensingCanvas.height;
    
    // Draw the lensed image (showing gravitational lensing effect)
    groundTruthCtx.drawImage(lensingCanvas, 0, 0);
    
    // Draw your dark matter markers
    markers.forEach(marker => {
        groundTruthCtx.beginPath();
        groundTruthCtx.arc(marker.x, marker.y, 12, 0, 2 * Math.PI);
        groundTruthCtx.fillStyle = 'rgba(255, 0, 0, 0.7)';
        groundTruthCtx.fill();
        groundTruthCtx.strokeStyle = 'darkred';
        groundTruthCtx.lineWidth = 3;
        groundTruthCtx.stroke();
        
        // Label
        groundTruthCtx.fillStyle = 'white';
        groundTruthCtx.font = 'bold 16px Arial';
        groundTruthCtx.fillText('YOU', marker.x - 12, marker.y - 25);
    });
}

async function runPrediction() {
    if (galaxyData.length === 0) {
        alert('Please show gravitational lensing effect first!');
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
        
        // Draw ground truth (your choices)
        drawGroundTruth();
        
        // Draw predictions
        drawPredictions();
        
        // Calculate accuracy
        const accuracy = calculateAccuracy();
        document.getElementById('accuracy').textContent = (accuracy * 100).toFixed(1) + '%';
        document.getElementById('predictedDarkMatter').textContent = 
            predictions.filter(p => p === 1).length;
        document.getElementById('darkMatterCount').textContent = markers.length;
        
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
    predictionCanvas.width = lensingCanvas.width;
    predictionCanvas.height = lensingCanvas.height;
    
    // Draw the lensed image (showing gravitational lensing)
    predictionCtx.drawImage(lensingCanvas, 0, 0);
    
    // Draw model predictions
    galaxyData.forEach((galaxy, index) => {
        if (index >= predictions.length) return;
        
        const prediction = predictions[index];
        predictionCtx.beginPath();
        predictionCtx.arc(galaxy.x, galaxy.y, 14, 0, 2 * Math.PI);
        
        if (prediction === 1) {
            // Predicted dark matter - yellow star
            predictionCtx.fillStyle = 'rgba(255, 255, 0, 0.6)';
            predictionCtx.fill();
            predictionCtx.strokeStyle = 'yellow';
            predictionCtx.lineWidth = 3;
            predictionCtx.stroke();
            
            // Draw star shape
            drawStar(predictionCtx, galaxy.x, galaxy.y, 5, 14, 5);
            
            // Label
            predictionCtx.fillStyle = 'white';
            predictionCtx.font = 'bold 14px Arial';
            predictionCtx.fillText('MODEL', galaxy.x - 20, galaxy.y - 25);
        } else {
            // Predicted background - light blue circle (smaller, less prominent)
            predictionCtx.fillStyle = 'rgba(173, 216, 230, 0.3)';
            predictionCtx.fill();
            predictionCtx.strokeStyle = 'cyan';
            predictionCtx.lineWidth = 1;
            predictionCtx.stroke();
        }
    });
    
    // Also draw your markers for comparison
    markers.forEach(marker => {
        predictionCtx.beginPath();
        predictionCtx.arc(marker.x, marker.y, 10, 0, 2 * Math.PI);
        predictionCtx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        predictionCtx.lineWidth = 2;
        predictionCtx.stroke();
    });
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

function loadExampleImage(type) {
    // Generate a synthetic galaxy field image on canvas
    const canvas = document.createElement('canvas');
    canvas.width = 1200;
    canvas.height = 800;
    const ctx = canvas.getContext('2d');
    
    // Create dark space background
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, '#0a0a0f');
    gradient.addColorStop(0.5, '#1a1a2e');
    gradient.addColorStop(1, '#16213e');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Add stars
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    for (let i = 0; i < 200; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        const size = Math.random() * 2;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
    }
    
    // Add galaxies - different patterns for each example
    const numGalaxies = type === 'example1' ? 30 : type === 'example2' ? 40 : 35;
    const galaxies = [];
    
    for (let i = 0; i < numGalaxies; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        const size = 10 + Math.random() * 25;
        const brightness = 0.3 + Math.random() * 0.7;
        const isDistorted = Math.random() > (type === 'example1' ? 0.6 : type === 'example2' ? 0.5 : 0.55);
        
        galaxies.push({x, y, size, brightness, isDistorted});
        
        // Draw galaxy
        const galaxyGradient = ctx.createRadialGradient(x, y, 0, x, y, size);
        galaxyGradient.addColorStop(0, `rgba(255, 255, 255, ${brightness})`);
        galaxyGradient.addColorStop(0.5, `rgba(200, 200, 255, ${brightness * 0.5})`);
        galaxyGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        
        ctx.fillStyle = galaxyGradient;
        
        // Draw elliptical galaxy (distorted if dark matter present)
        ctx.save();
        ctx.translate(x, y);
        if (isDistorted) {
            // Distorted ellipse (gravitational lensing effect)
            ctx.scale(1.3, 0.7);
            ctx.rotate(Math.random() * Math.PI / 4);
        } else {
            // Normal ellipse
            ctx.scale(1, 0.8);
            ctx.rotate(Math.random() * Math.PI);
        }
        ctx.beginPath();
        ctx.ellipse(0, 0, size, size * 0.6, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        
        // Add core
        ctx.fillStyle = `rgba(255, 255, 200, ${brightness})`;
        ctx.beginPath();
        ctx.arc(x, y, size * 0.3, 0, Math.PI * 2);
        ctx.fill();
    }
    
    // Convert canvas to image and load it
    canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
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
            document.getElementById('exampleImagesGrid').style.display = 'none';
            
            // Store original image data for lensing effect
            originalImageData = imageCtx.getImageData(0, 0, width, height);
            
            // Store galaxy positions
            window.exampleGalaxies = galaxies;
            
            URL.revokeObjectURL(url);
        };
        img.src = url;
    });
}

function resetAll() {
    markers = [];
    galaxyData = [];
    predictions = [];
    originalImageData = null;
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('lensingSection').style.display = 'none';
    document.getElementById('canvasContainer').style.display = 'none';
    document.getElementById('controlsPanel').style.display = 'none';
    document.getElementById('exampleImagesGrid').style.display = 'grid';
    document.getElementById('showLensingBtn').style.display = 'none';
    document.getElementById('predictBtn').style.display = 'none';
    document.querySelectorAll('.example-image-card').forEach(c => c.classList.remove('selected'));
    imageFile = null;
    updateStats();
}

