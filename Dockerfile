FROM python:3.11-slim

# Instalar curl para descargar modelos
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/

# Descargar modelos desde Hugging Face Hub
RUN curl -L -o model_weights.pt https://huggingface.co/pagredadiaz/clasificacion-documentos-modelos/resolve/main/model_weights.pt
RUN curl -L -o model_artifacts.pkl https://huggingface.co/pagredadiaz/clasificacion-documentos-modelos/resolve/main/model_artifacts.pkl

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]