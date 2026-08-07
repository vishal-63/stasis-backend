FROM python:3.12-slim

# Don't run as root
RUN useradd --create-home appuser

WORKDIR /home/appuser/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN node --version

# Install Python deps as root before switching user
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade yt-dlp

# Copy app code
COPY app/ ./app/

# Switch to non-root user
USER appuser

EXPOSE 8000

# Run with a single worker on Render free tier (512MB RAM)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]