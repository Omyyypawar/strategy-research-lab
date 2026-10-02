FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and supporting directories
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY tests/ tests/

# Copy env template (never copy .env with real secrets)
COPY .env.example .env.example

# Set Python path so imports resolve from project root
ENV PYTHONPATH=/app

# Default: run the live executor
CMD ["python", "-m", "src.live_executor"]
