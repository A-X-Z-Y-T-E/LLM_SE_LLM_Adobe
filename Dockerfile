FROM python:3.10-slim

# Ensure all system packages are up-to-date to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y && apt-get clean

# Set working directory
WORKDIR /app

# Install system dependencies required for PyTorch and PyMuPDF
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel for better package management
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with explicit CPU-only PyTorch 2.5.0
RUN pip install --no-cache-dir torch==2.5.0+cpu torchvision==0.20.0+cpu torchaudio==2.5.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-geometric==2.5.0
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY process_pdfs.py .
COPY complete_pdf_to_outline_pipeline.py .
COPY extractor/ ./extractor/
COPY model_training/ ./model_training/


# Copy the trained model
COPY updated_model_8.pth .

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Set environment variables for CPU-only execution
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_HOME=/tmp/torch
ENV OMP_NUM_THREADS=8

# Verify PyTorch CPU installation and model compatibility
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CPU threads: {torch.get_num_threads()}')"

# Test model loading capability
RUN python -c "import torch; model_data = torch.load('updated_model_8.pth', map_location='cpu'); print('✅ V8 model loads successfully on CPU')" || echo "⚠️ V8 model not found - will use fallback"

# Run the PDF processing script
CMD ["python", "./process_pdfs.py"]
