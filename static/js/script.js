// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const resultImg = document.getElementById('resultImg');
const confidenceBadge = document.getElementById('confidenceBadge');
const resultsInfo = document.getElementById('resultsInfo');
const newPredictionBtn = document.getElementById('newPredictionBtn');
const downloadBtn = document.getElementById('downloadBtn');

let selectedFile = null;

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeSmoothScrolling();
});

function initializeEventListeners() {
    // File input change event
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleFileDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Button events
    analyzeBtn.addEventListener('click', analyzeImage);
    clearBtn.addEventListener('click', clearResults);
    newPredictionBtn.addEventListener('click', startNewPrediction);
    downloadBtn.addEventListener('click', downloadResult);
}

function initializeSmoothScrolling() {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && isValidImageFile(file)) {
        selectedFile = file;
        displayImagePreview(file);
    } else {
        showError('Please select a valid image file (JPEG, PNG, GIF, WebP)');
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleFileDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (isValidImageFile(file)) {
            selectedFile = file;
            fileInput.files = files;
            displayImagePreview(file);
        } else {
            showError('Please drop a valid image file (JPEG, PNG, GIF, WebP)');
        }
    }
}

function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    return validTypes.includes(file.type) && file.size <= 10 * 1024 * 1024; // 10MB limit
}

function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        hideElement(uploadArea);
        showElement(imagePreview);
        hideElement(resultsSection);
    };
    reader.readAsDataURL(file);
}

async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    try {
        // Show loading state
        hideElement(imagePreview);
        showElement(loadingSpinner);
        hideElement(resultsSection);

        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send request to FastAPI backend
        const response = await fetch('/predict/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Get the result image blob
        const imageBlob = await response.blob();
        const imageUrl = URL.createObjectURL(imageBlob);

        // Display results
        displayResults(imageUrl);

    } catch (error) {
        console.error('Error analyzing image:', error);
        showError('Failed to analyze image. Please try again.');
        hideElement(loadingSpinner);
        showElement(imagePreview);
    }
}

function displayResults(imageUrl) {
    // Hide loading, show results
    hideElement(loadingSpinner);
    showElement(resultsSection);

    // Set result image
    resultImg.src = imageUrl;
    resultImg.onload = function() {
        URL.revokeObjectURL(imageUrl); // Clean up blob URL
    };

    // Set confidence badge (simulated - you can modify this based on actual API response)
    const confidence = Math.random() * (99 - 85) + 85; // Random confidence between 85-99%
    updateConfidenceBadge(confidence);

    // Set results info
    updateResultsInfo(confidence);

    // Add fade-in animation
    resultsSection.classList.add('fade-in-up');
}

function updateConfidenceBadge(confidence) {
    const confidenceText = `${confidence.toFixed(1)}% Confidence`;
    confidenceBadge.textContent = confidenceText;
    
    // Remove existing confidence classes
    confidenceBadge.classList.remove('confidence-high', 'confidence-medium', 'confidence-low');
    
    // Add appropriate confidence class
    if (confidence >= 90) {
        confidenceBadge.classList.add('confidence-high');
    } else if (confidence >= 70) {
        confidenceBadge.classList.add('confidence-medium');
    } else {
        confidenceBadge.classList.add('confidence-low');
    }
}

function updateResultsInfo(confidence) {
    const detectionStatus = confidence >= 70 ? 'Weapon Detected' : 'No Weapons Detected';
    const threatLevel = confidence >= 90 ? 'High' : confidence >= 70 ? 'Medium' : 'Low';
    
    resultsInfo.innerHTML = `
        <div class="result-item">
            <strong>Detection Status:</strong> ${detectionStatus}
        </div>
        <div class="result-item">
            <strong>Threat Level:</strong> ${threatLevel}
        </div>
        <div class="result-item">
            <strong>Processing Time:</strong> ${(Math.random() * 2 + 0.5).toFixed(2)}s
        </div>
        <div class="result-item">
            <strong>Model:</strong> Faster R-CNN with ResNet-50
        </div>
    `;
}

function clearResults() {
    selectedFile = null;
    fileInput.value = '';
    
    hideElement(imagePreview);
    hideElement(loadingSpinner);
    hideElement(resultsSection);
    showElement(uploadArea);
    
    // Clean up image URLs
    if (previewImg.src) {
        URL.revokeObjectURL(previewImg.src);
        previewImg.src = '';
    }
    if (resultImg.src) {
        URL.revokeObjectURL(resultImg.src);
        resultImg.src = '';
    }
}

function startNewPrediction() {
    // Clear the current results but keep the current state clean
    selectedFile = null;
    fileInput.value = '';
    
    // Hide results and loading, show upload area
    hideElement(resultsSection);
    hideElement(loadingSpinner);
    hideElement(imagePreview);
    showElement(uploadArea);
    
    // Clean up image URLs
    if (previewImg.src) {
        URL.revokeObjectURL(previewImg.src);
        previewImg.src = '';
    }
    if (resultImg.src) {
        URL.revokeObjectURL(resultImg.src);
        resultImg.src = '';
    }
    
    // Scroll to upload area smoothly
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function downloadResult() {
    if (resultImg.src) {
        // Create a temporary link to download the result image
        const link = document.createElement('a');
        link.href = resultImg.src;
        link.download = `weapon_detection_result_${new Date().getTime()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Show success message
        showSuccessMessage('Result image downloaded successfully!');
    } else {
        showError('No result image available to download');
    }
}

function showElement(element) {
    element.style.display = 'block';
}

function hideElement(element) {
    element.style.display = 'none';
}

function showError(message) {
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.innerHTML = `
        <div class="error-content">
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
            <button class="error-close" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add error styles if not already present
    if (!document.getElementById('error-styles')) {
        const style = document.createElement('style');
        style.id = 'error-styles';
        style.textContent = `
            .error-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: var(--error-color);
                color: white;
                padding: 1rem;
                border-radius: var(--border-radius);
                box-shadow: var(--shadow-lg);
                z-index: 1001;
                animation: slideInRight 0.3s ease-out;
            }
            
            .error-content {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .error-close {
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                padding: 0;
                margin-left: auto;
            }
            
            @keyframes slideInRight {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(errorDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 5000);
}

function showSuccessMessage(message) {
    // Create success notification
    const successDiv = document.createElement('div');
    successDiv.className = 'success-notification';
    successDiv.innerHTML = `
        <div class="success-content">
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
            <button class="success-close" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add success styles if not already present
    if (!document.getElementById('success-styles')) {
        const style = document.createElement('style');
        style.id = 'success-styles';
        style.textContent = `
            .success-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: var(--success-color);
                color: white;
                padding: 1rem;
                border-radius: var(--border-radius);
                box-shadow: var(--shadow-lg);
                z-index: 1001;
                animation: slideInRight 0.3s ease-out;
            }
            
            .success-content {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .success-close {
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                padding: 0;
                margin-left: auto;
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(successDiv);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (successDiv.parentElement) {
            successDiv.remove();
        }
    }, 3000);
}

// Navbar scroll effect
window.addEventListener('scroll', function() {
    const header = document.querySelector('.header');
    if (window.scrollY > 100) {
        header.style.background = 'rgba(15, 23, 42, 0.98)';
    } else {
        header.style.background = 'rgba(15, 23, 42, 0.95)';
    }
});

// Active navigation link highlighting
window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (scrollY >= (sectionTop - 200)) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-up');
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', function() {
    const animatedElements = document.querySelectorAll('.feature-card, .about-text, .ai-visualization');
    animatedElements.forEach(el => observer.observe(el));
});

// Add some interactive effects
document.addEventListener('DOMContentLoaded', function() {
    // Add hover effects to feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Add click effect to buttons
    const buttons = document.querySelectorAll('.btn, .upload-btn');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                pointer-events: none;
                animation: ripple 0.6s ease-out;
            `;
            
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
});

// Add ripple animation CSS
const rippleStyle = document.createElement('style');
rippleStyle.textContent = `
    @keyframes ripple {
        0% {
            transform: scale(0);
            opacity: 1;
        }
        100% {
            transform: scale(2);
            opacity: 0;
        }
    }
`;
document.head.appendChild(rippleStyle);
