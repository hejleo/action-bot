# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create HuggingFace cache directory
RUN mkdir -p /root/.cache/huggingface

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "-m", "incar.interface"]