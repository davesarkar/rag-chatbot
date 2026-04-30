FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

RUN mkdir -p documents chroma_db

EXPOSE 8080

# Use shell form so $PORT gets expanded
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}