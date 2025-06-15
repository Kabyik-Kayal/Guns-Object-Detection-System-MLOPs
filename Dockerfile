# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y

# Copy requirements file
COPY docker-requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r docker-requirements.txt

# Create necessary directories
RUN mkdir -p src artifacts/models static

# Copy specific source files
COPY src/model_architecture.py src/
COPY src/custom_exception.py src/
COPY src/logger.py src/
COPY src/data_processing.py src/
COPY static/ static/
COPY main.py .
COPY run_server.py .

# Copy the trained model
COPY artifacts/models/ artifacts/models/

# Create empty __init__.py files for Python modules
RUN touch src/__init__.py

# Download the pretrained model weights
RUN python -c "import torch; import torchvision; torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')"

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["python", "run_server.py"]