FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

WORKDIR /app




