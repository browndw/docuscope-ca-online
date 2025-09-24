FROM python:3.11-slim

# Install system dependencies needed for building Python packages
# and runtime libraries required by Plotly Kaleido (headless image export)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    # Kaleido/Chromium and font rendering deps
    libexpat1 \
    libglib2.0-0 \
    libnss3 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxi6 \
    libxss1 \
    libxtst6 \
    libx11-xcb1 \
    libxcb1 \
    libxshmfence1 \
    libxfixes3 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    fonts-liberation \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash streamlit

# Set working directory to project root
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R streamlit:streamlit /app

# Switch to non-root user
USER streamlit

# Expose port
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application
CMD ["streamlit", "run", "webapp/index.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
