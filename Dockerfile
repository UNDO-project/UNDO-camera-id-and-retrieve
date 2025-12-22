# ==============================================================================
# Stage 1: Builder - Install dependencies using uv
# ==============================================================================
FROM python:3.12-slim AS builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory for build
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Sync dependencies using uv (creates .venv in /app)
# --frozen ensures we use exact versions from uv.lock
# --no-dev excludes development dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ==============================================================================
# Stage 2: Runtime - Minimal production image
# ==============================================================================
FROM python:3.12-slim AS runtime

# Install system dependencies required by PyTorch, PIL, and CLIP
# - libgomp1: OpenMP library for PyTorch CPU operations
# - libglib2.0-0: Required by PIL for image processing
# - libsm6, libxext6, libxrender1: X11 libraries for OpenCV (if needed)
# - libgl1: OpenGL library for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# UID 1000 is standard for first user in many systems
RUN useradd --create-home --uid 1000 --shell /bin/bash cidar

# Set working directory
WORKDIR /app

# Copy Python virtual environment from builder stage
COPY --from=builder --chown=cidar:cidar /app/.venv /app/.venv

# Copy application source code
COPY --chown=cidar:cidar src/ /app/src/
COPY --chown=cidar:cidar pyproject.toml /app/

# Create necessary directories with proper permissions
# - output: for parquet, embeddings, and query_patches
# - model_weights: for YOLO weights
# - data: for scraped data (if needed)
RUN mkdir -p /app/output /app/model_weights /app/data && \
    chown -R cidar:cidar /app/output /app/model_weights /app/data

# Set environment variables for runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    # Force CPU-only for PyTorch
    CUDA_VISIBLE_DEVICES="" \
    # Set project root for PathSettings auto-detection
    CIDAR_PATH_PROJECT_ROOT=/app

# Switch to non-root user
USER cidar

# Expose API port
EXPOSE 8000

# Health check using the API health endpoint
# - interval: how often to check (30s)
# - timeout: how long to wait for response (10s)
# - start-period: grace period before first check (60s for model downloads)
# - retries: number of failures before marking unhealthy (3)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/health', timeout=5.0).raise_for_status()"

# Default command: run the API server
# Using uvicorn directly (not the cidar-api script) for better container control
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]