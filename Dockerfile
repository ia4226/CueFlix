FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy app code
COPY app/ ./app/
COPY data/enriched_data.csv ./data/enriched_data.csv
COPY data/index.pkl ./data/index.pkl

# Expose port 80
EXPOSE 80

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]