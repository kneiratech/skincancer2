let model;
let isModelLoaded = false;
let isCameraActive = false;
let stream = null;

const l2Regularizer = (lambda) => {
  return tf.regularizers.l2(lambda);
};

// Class names for the model predictions
const classNames = ['Benign', 'Malignant'];

// Load the TensorFlow model
async function loadModel() {
    try {
        
        updateStatus('Loading Model...', 'loading');
        
        // For demo purposes, we'll simulate model loading
        // In a real implementation, you would load your actual model:
        model = await tf.loadLayersModel('model.json');
        
        // Simulate loading time
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Create a dummy model for demonstration
        model = await createDummyModel();
        
        isModelLoaded = true;
        updateStatus('Model Ready', 'ready');
        enableButtons();
        console.log("Model Loaded Successfully!");
    } catch (error) {
        console.error("Error loading model:", error);
        updateStatus('Model Load Failed', 'error');
        document.getElementById('predictionResult').textContent = 'Error loading model. Please refresh the page.';
    }
}


function updateStatus(message, type) {
    const statusEl = document.getElementById('modelStatus');
    statusEl.className = `status-indicator status-${type}`;
    statusEl.innerHTML = type === 'loading' ? 
        `<span class="loading"></span> ${message}` : 
        message;
    
    if (type === 'ready') {
        setTimeout(() => {
            statusEl.style.opacity = '0';
            setTimeout(() => statusEl.style.display = 'none', 300);
        }, 2000);
    }
}

function enableButtons() {
    document.getElementById('uploadBtn').disabled = false;
    document.getElementById('cameraBtn').disabled = false;
}

function triggerFileInput() {
    document.getElementById('imageInput').click();
}

// Handle image upload
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    stopCamera();

    const img = new Image();
    img.onload = function() {
        displayImage(img);
        processImage(img);
    }
    img.src = URL.createObjectURL(file);
}

function displayImage(img) {
    const previewImg = document.getElementById('previewImage');
    const placeholder = document.getElementById('placeholderText');
    const container = document.getElementById('imageContainer');
    const webcam = document.getElementById('webcam');

    webcam.classList.add('hidden');
    placeholder.classList.add('hidden');
    previewImg.src = img.src;
    previewImg.classList.remove('hidden');
    container.classList.add('has-image');
}

// Toggle camera functionality
async function toggleCamera() {
    if (isCameraActive) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    try {
        const video = document.getElementById('webcam');
        const placeholder = document.getElementById('placeholderText');
        const previewImg = document.getElementById('previewImage');
        const container = document.getElementById('imageContainer');
        const cameraBtn = document.getElementById('cameraBtn');

        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 300, 
                height: 300,
                facingMode: 'environment' // Use back camera if available
            } 
        });
        
        video.srcObject = stream;
        video.play();

        placeholder.classList.add('hidden');
        previewImg.classList.add('hidden');
        video.classList.remove('hidden');
        container.classList.add('has-image');

        isCameraActive = true;
        cameraBtn.textContent = '⏹️ Stop Camera';

        // Capture and analyze image every 2 seconds
        const captureInterval = setInterval(() => {
            if (isCameraActive) {
                captureAndAnalyze();
            } else {
                clearInterval(captureInterval);
            }
        }, 2000);

    } catch (error) {
        console.error("Error accessing camera:", error);
        alert("Unable to access camera. Please ensure you have granted camera permissions.");
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    const video = document.getElementById('webcam');
    const cameraBtn = document.getElementById('cameraBtn');
    
    video.classList.add('hidden');
    isCameraActive = false;
    cameraBtn.textContent = '📷 Use Camera';

    // Reset to placeholder if no image is shown
    const previewImg = document.getElementById('previewImage');
    if (previewImg.classList.contains('hidden')) {
        resetToPlaceholder();
    }
}

function captureAndAnalyze() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = 300;
    canvas.height = 300;
    ctx.drawImage(video, 0, 0, 300, 300);
    
    canvas.toBlob(blob => {
        const img = new Image();
        img.onload = function() {
            processImage(img);
        }
        img.src = URL.createObjectURL(blob);
    }, 'image/jpeg');
}

function resetToPlaceholder() {
    const placeholder = document.getElementById('placeholderText');
    const container = document.getElementById('imageContainer');
    
    placeholder.classList.remove('hidden');
    container.classList.remove('has-image');
}

// Process the image and make a prediction
async function processImage(image) {
    if (!isModelLoaded) {
        document.getElementById('predictionResult').textContent = 'Model is still loading. Please wait...';
        return;
    }

    try {
        // Show loading state
        document.getElementById('predictionResult').innerHTML = 
            '<span class="loading"></span> Analyzing image...';

        // Preprocess the image
        const imgTensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(255.0)
            .expandDims(0);

        // Make prediction
        const prediction = await model.predict(imgTensor);
        const predictionData = await prediction.data();
        
        // Clean up tensors
        imgTensor.dispose();
        prediction.dispose();

        // Display results
        displayPrediction(predictionData[0]);

    } catch (error) {
        console.error("Error making prediction:", error);
        document.getElementById('predictionResult').textContent = 
            'Error analyzing image. Please try again.';
    }
}

// Display the prediction result
function displayPrediction(confidence) {
    const resultEl = document.getElementById('predictionResult');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceFill = document.getElementById('confidenceFill');

    // For binary classification: >0.5 = Malignant, <=0.5 = Benign
    const isMalignant = confidence > 0.5;
    const displayConfidence = isMalignant ? confidence : 1 - confidence;
    const prediction = isMalignant ? 'Malignant' : 'Benign';
    
    // Update result text and styling
    resultEl.className = `prediction-result result-${prediction.toLowerCase()}`;
    resultEl.innerHTML = `
        <div style="font-size: 1.5rem; margin-bottom: 10px;">
            ${prediction === 'Malignant' ? '⚠️' : '✅'} ${prediction}
        </div>
        <div style="font-size: 1rem; opacity: 0.8;">
            Confidence: ${(displayConfidence * 100).toFixed(1)}%
        </div>
    `;

    // Update confidence bar
    confidenceBar.classList.remove('hidden');
    confidenceFill.style.width = `${displayConfidence * 100}%`;
    confidenceFill.style.background = isMalignant ? 
        'linear-gradient(45deg, #fc8181, #e53e3e)' : 
        'linear-gradient(45deg, #68d391, #38a169)';
}

// Initialize the application
window.onload = function() {
    loadModel();
};

// Cleanup on page unload
window.onbeforeunload = function() {
    stopCamera();
};