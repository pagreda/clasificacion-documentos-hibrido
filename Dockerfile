FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/

# Descargar modelos desde Hugging Face Hub
RUN curl -L -o app/model_weights.pt https://huggingface.co/pagredadiaz/clasificacion-documentos-modelos/blob/main/model_weights.pt
RUN curl -L -o app/model_artifacts.pkl https://huggingface.co/pagredadiaz/clasificacion-documentos-modelos/blob/main/model_artifacts.pkl

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]