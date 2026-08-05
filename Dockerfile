FROM python:3.12-slim

# Don't run as root
RUN useradd --create-home appuser

WORKDIR /home/appuser/app

# Install dependencies needed to download and extract the static build
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Download the yt-dlp specific FFmpeg master build, extract it, and move binaries to /usr/local/bin
RUN wget https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
    && tar -xf ffmpeg-master-latest-linux64-gpl.tar.xz \
    && mv ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ \
    && mv ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ \
    && rm -rf ffmpeg-master-latest-linux64-gpl*

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