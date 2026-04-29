FROM python:3.11-slim

WORKDIR /app

# Copy from backend/ folder
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ .

# Create required folders
RUN mkdir -p documents chroma_db

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]