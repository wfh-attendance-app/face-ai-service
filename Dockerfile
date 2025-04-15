FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./app /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    face_recognition \
    faiss-cpu \
    pillow \
    opencv-python \
    python-multipart
