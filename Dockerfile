# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/sisazac23/geographical-visualization-platform

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD streamlit run Home.py