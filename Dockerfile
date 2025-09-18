FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv python3-dev build-essential git curl wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
RUN pip3 install --no-cache-dir .

# Ensure target directories for volumes exist
RUN mkdir -p /app/outputs /app/checkpoints /app/logs && \
    chown -R 1001:1001 /app/outputs /app/checkpoints /app/logs

# Change ownership of app files to non-root user
RUN chown -R 1001:1001 /app

# Create a non-root user to run the application
USER 1001

# Expose the port the app runs on
EXPOSE 8080

VOLUME [ "/app/checkpoints", "/app/outputs", "/app/logs" ]

# Set healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=5 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application with GPU support
CMD ["python3", "api.py"]
