# 🌐 GuardAI Web Application

A beautiful, modern web interface for the Guns Object Detection System, powered by FastAPI and advanced AI.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python run_server.py
   ```
   
   Or alternatively:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Open Your Browser**
   Navigate to `http://localhost:8000` to access the web application.

## 🎯 Features

- **🖱️ Drag & Drop Interface**: Simply drag and drop images for instant analysis
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **🎨 Modern UI**: Beautiful, dark-themed interface with smooth animations
- **⚡ Real-time Processing**: Fast AI-powered weapon detection
- **📊 Confidence Scoring**: Visual confidence indicators for detection results
- **🔄 Interactive Results**: Clear visualization of detected objects with bounding boxes

## 🏗️ Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with automatic API documentation
- **AI Model**: Faster R-CNN with ResNet-50 backbone
- **Image Processing**: PyTorch and PIL for image manipulation
- **CORS Support**: Cross-origin resource sharing enabled

### Frontend
- **Technology**: Vanilla JavaScript, HTML5, CSS3
- **Design**: Modern dark theme with gradient accents
- **Animations**: Smooth transitions and hover effects
- **Icons**: Font Awesome for consistent iconography

## 📁 Project Structure

```
static/
├── index.html          # Main web application
├── css/
│   └── style.css      # Modern styling and animations
└── js/
    └── script.js      # Interactive functionality

main.py                # FastAPI backend server
run_server.py         # Server startup script
```

## 🛠️ API Endpoints

### `GET /`
- **Description**: Serves the main web application
- **Response**: HTML page with the full interface

### `POST /predict/`
- **Description**: Analyze uploaded image for weapon detection
- **Parameters**: 
  - `file`: Image file (JPEG, PNG, GIF, WebP)
- **Response**: Processed image with bounding boxes around detected objects

## 🎨 Design Features

### Color Scheme
- **Primary**: Indigo blue (#6366f1)
- **Secondary**: Emerald green (#10b981)
- **Background**: Dark slate (#0f172a, #1e293b)
- **Text**: Light gray tones for optimal readability

### Interactive Elements
- **Hover Effects**: Smooth transitions on cards and buttons
- **Loading States**: Elegant spinners during processing
- **Drag & Drop**: Visual feedback for file uploads
- **Animations**: Fade-in effects and neural network visualization

## 🔧 Configuration

### Model Settings
The application automatically loads the trained model from `artifacts/models/fasterrcnn.pth` with:
- **Confidence Threshold**: 0.7 (70%)
- **Device**: Auto-detection (CUDA if available, otherwise CPU)
- **Classes**: 2 (background, weapon)

### Server Settings
- **Host**: 0.0.0.0 (accessible from all network interfaces)
- **Port**: 8000
- **Reload**: Enabled for development
- **CORS**: Enabled for all origins

## 📱 Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 🚀 Deployment

### Local Development
```bash
python run_server.py
```

### Production Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
The application includes a Dockerfile for containerized deployment.

## 🎯 Usage Tips

1. **Supported Formats**: JPEG, PNG, GIF, WebP (max 10MB)
2. **Optimal Images**: Clear, well-lit images work best
3. **Processing Time**: Typically 1-3 seconds depending on image size
4. **Results**: Red bounding boxes indicate detected weapons
5. **Confidence**: Higher confidence scores indicate more certain detections

## 🔒 Security Features

- **File Validation**: Only image files are accepted
- **Size Limits**: 10MB maximum file size
- **CORS Protection**: Configurable cross-origin settings
- **Input Sanitization**: Safe file handling and processing

## 🎨 Customization

### Themes
The CSS variables in `static/css/style.css` can be modified to change:
- Color scheme
- Fonts
- Spacing
- Border radius
- Shadows

### Confidence Thresholds
Modify the confidence threshold in `main.py`:
```python
if score > 0.7:  # Change this value
```

## 📈 Performance

- **Model Loading**: One-time initialization on startup
- **Inference Speed**: ~1-2 seconds per image
- **Memory Usage**: Optimized for production deployment
- **Concurrent Requests**: Supported via FastAPI async handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Guns Object Detection System MLOps pipeline.

---

**🛡️ GuardAI - Advanced AI Security Solutions**
