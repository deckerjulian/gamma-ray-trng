# Multi-stage build for production
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with specific UID/GID to match host
RUN groupadd -g 1000 app && \
    useradd -u 1000 -g 1000 --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy application code
COPY backend.py .
COPY nist_randomness_tests.py .

# Copy entire static directory with all subdirectories and files
COPY static/ ./static/

# Create instance directory with proper permissions
RUN mkdir -p /app/instance && \
    touch /app/instance/random_bits.txt && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Add local Python packages to PATH
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/home/app/.local/lib/python3.11/site-packages

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=10)" || exit 1

# Run the application
CMD ["python", "-m", "flask", "--app", "backend", "run", "--host", "0.0.0.0", "--port", "5000"]