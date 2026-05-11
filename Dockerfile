FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by spaCy and BeautifulSoup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Dutch spaCy model
RUN python -m spacy download nl_core_news_lg

# Copy source files
COPY preprocess.py .
COPY config.py .

# Output directory (will be overridden by volume mount)
RUN mkdir -p /app/output /app/data /app/notebooks
