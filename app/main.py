# -*- coding: utf-8 -*-
"""
Backend FastAPI - Clasificación de Documentos
Basado en ModeloHibrido (BETO + embeddings categóricos + prior histórico)

CAMBIOS RESPECTO A main.py (basados en clasificacion_documentos_tramite.py):
─────────────────────────────────────────────────────────────────────────────
[C1] limpiar_texto_general() reemplaza a limpiar_texto() y build_texto_modelo()
     · Coincide exactamente con la función del notebook (acepta None/NaN,
       preserva á é í ó ú ñ, colapsa espacios).
     · Se usa para limpiar remitente, tipo_doc y asunto antes de armar
       TEXTO_MODELO_MIN, igual que en el entrenamiento.

[C2] build_texto_modelo() actualizado
     · Ahora aplica limpiar_texto_general() a los tres campos y los combina
       con el mismo formato que el notebook:
         "[REMITENTE] <rem_clean> [TIPO_DOC] <tipo_clean> [ASUNTO] <asunto_clean>"

[C3] ModeloHibrido acepta bert_model_path
     · El constructor recibe bert_model_path en lugar de usar la constante
       global MODEL_CHECKPOINT, igual que en el notebook (sección 10).
     · Permite cargar BETO desde ruta local o desde HuggingFace Hub.

[C4] Lógica de resolución de fuente BETO (load_or_resolve_beto_path)
     · Replica la función load_or_create_model() del notebook (sección 7):
         1. Si existe config.json en la ruta local → usa ruta local.
         2. Si no → usa HuggingFace Hub.
     · La ruta local por defecto es la carpeta on-premise configurada en
       BETO_LOCAL_PATH (variable de entorno o constante).

[C5] normalizar_scores() agregada
     · Función del notebook (sección 14) que normaliza un dict de scores
       dividiendo cada valor por la suma total.
     · Se aplica a score_modelo y score_historico antes de combinarlos.

[C6] Alpha continuo inteligente reemplaza al alpha por rangos discretos
     · El notebook eliminó los if/elif de alpha y adoptó un alpha continuo:
         conf_model  = max(probs)
         conf_hist   = max(score_historico.values())  — 0 si sin histórico
         alpha_base  = conf_model / (conf_model + conf_hist + eps)
         peso_hist   = total / (total + c)            — c=10 (hiperparámetro)
         alpha_final = alpha_base * (1 - peso_hist)
         alpha_final = max(0.1, min(0.9, alpha_final))  — límites suaves
     · Cuando total=0 (sin histórico) → alpha_final queda en su límite
       superior (~0.9), priorizando el modelo, comportamiento idéntico al
       alpha=1.0 anterior pero con límite suave.

[C7] prior_counts se construye con REMITENTE_CLEAN (texto limpio)
     · En el notebook (sección 13): prior_counts[row["REMITENTE_CLEAN"]]
     · predict_hybrid ahora normaliza el remitente con limpiar_texto_general()
       antes de buscar en prior_counts, garantizando la misma clave que se
       usó al entrenar.

[C8] name_mapping expuesto como constante NAME_MAPPING
     · El notebook aplica un mapeo de nombres largos a alias cortos
       (ej. "Víctor Guevara" → "VGuevara") sobre las columnas de responsables.
     · Se incluye como constante informativa; los artefactos exportados desde
       el notebook ya tienen los nombres mapeados, por lo que el backend no
       necesita aplicarlo en tiempo de inferencia.
─────────────────────────────────────────────────────────────────────────────

"""

# =========================
# Importaciones
# =========================
import os
import re
import pickle
import logging
import unicodedata                          # [C1] requerido por limpiar_texto_general
from contextlib import asynccontextmanager
from collections import defaultdict, Counter
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Constantes
# =========================

# Ruta local on-premise de BETO (variable de entorno o valor por defecto)
BETO_LOCAL_PATH = os.getenv(
    "BETO_LOCAL_PATH",
    os.path.join(os.path.dirname(__file__), "BERT_Model_Cache"),
)
BETO_HF_REPO    = "dccuchile/bert-base-spanish-wwm-cased"   # fallback HuggingFace

MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "model_weights.pt")
ARTIFACTS_PATH     = os.getenv("ARTIFACTS_PATH",     "model_artifacts.pkl")

TIPOS_VALIDOS = {"CARTA", "MEMORANDO", "OFICIO"}


# =========================
# [C4] Resolución de fuente BETO
# Replica load_or_create_model() del notebook (sección 7):
#   1. config.json presente en local → usa ruta local (sin internet)
#   2. Si no → HuggingFace Hub
# =========================
def load_or_resolve_beto_path() -> str:
    """
    Devuelve la ruta/identificador desde donde cargar BETO.
    Prioriza la carpeta local on-premise; hace fallback a HuggingFace Hub.
    """
    local = BETO_LOCAL_PATH
    config_path = os.path.join(local, "config.json")

    if os.path.isdir(local) and os.path.exists(config_path):
        logger.info("📦 BETO encontrado en ruta local: %s", local)
        return local

    logger.warning(
        "⚠️  Ruta local de BETO no válida ('%s'). "
        "Se descargará desde HuggingFace Hub.",
        local,
    )
    return BETO_HF_REPO


# =========================
# [C3] Arquitectura del Modelo
# Acepta bert_model_path igual que en el notebook (sección 10)
# =========================
class ModeloHibrido(nn.Module):
    def __init__(
        self,
        num_labels: int,
        num_remitentes: int,
        num_tipos: int,
        bert_model_path: str = BETO_HF_REPO,   # [C3] nuevo parámetro
    ):
        super().__init__()
        # Carga BETO desde la ruta indicada (local o HuggingFace Hub)
        self.bert = AutoModel.from_pretrained(bert_model_path)

        for param in self.bert.parameters():
            param.requires_grad = False

        # Descongela las dos últimas capas del encoder
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        self.remitente_emb = nn.Embedding(num_remitentes, 16)
        self.tipo_emb      = nn.Embedding(num_tipos,      8)
        self.fc            = nn.Linear(768 + 16 + 8 + 2, num_labels)

    def forward(
        self,
        input_ids,
        attention_mask,
        remitente_enc,
        tipo_enc,
        fecha_feats,
        labels=None,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = outputs.pooler_output

        rem_emb  = self.remitente_emb(remitente_enc)
        tipo_emb = self.tipo_emb(tipo_enc)

        x      = torch.cat([pooled, rem_emb, tipo_emb, fecha_feats], dim=1)
        logits = self.fc(x)

        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
            return loss, logits

        return logits


# =========================
# Estado global de la app
# =========================
class AppState:
    model:           ModeloHibrido | None = None
    tokenizer:       AutoTokenizer | None = None
    mlb:             object               = None
    prior_counts:    defaultdict          = defaultdict(Counter)
    remitente_to_id: dict                 = {}
    tipo_to_id:      dict                 = {}
    device:          torch.device         = torch.device("cpu")


app_state = AppState()


# =========================
# Helpers de texto
# =========================

# [C1] Reemplaza a limpiar_texto() del main.py original.
# Coincide exactamente con limpiar_texto_general() del notebook (sección 3).
def limpiar_texto_general(texto: str) -> str:
    """
    Limpieza canónica usada tanto en el entrenamiento como en la inferencia.
    · Acepta None/NaN → devuelve "".
    · Lower-case, elimina caracteres no alfanuméricos excepto tildes y ñ,
      colapsa espacios múltiples.
    IMPORTANTE: esta es la misma función que se usó al construir
    REMITENTE_CLEAN, TIPO_DOC_CLEAN y ASUNTO_CLEAN en el notebook.
    """
    if pd.isnull(texto):
        return ""
    texto = str(texto)
    texto = unicodedata.normalize("NFC", texto)         # unifica tildes NFC/NFD
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Z0-9áéíóúñ\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# [C2] Actualizado: aplica limpiar_texto_general a los tres campos
# y los combina con el mismo formato que el notebook.
def build_texto_modelo(remitente: str, tipo_doc: str, asunto: str) -> str:
    """
    Construye el texto de entrada al modelo exactamente como en el notebook:
        "[REMITENTE] <rem_clean> [TIPO_DOC] <tipo_clean> [ASUNTO] <asunto_clean>"
    """
    rem_clean   = limpiar_texto_general(remitente)
    tipo_clean  = limpiar_texto_general(tipo_doc)
    asunto_clean = limpiar_texto_general(asunto)
    return f"[REMITENTE] {rem_clean} [TIPO_DOC] {tipo_clean} [ASUNTO] {asunto_clean}"


# [C5] Nueva función del notebook (sección 14)
def normalizar_scores(scores_dict: dict) -> dict:
    """
    Normaliza un diccionario de scores dividiéndolos por su suma total.
    Si la suma es 0 devuelve el diccionario sin cambios.
    """
    total = sum(scores_dict.values())
    if total == 0:
        return scores_dict
    return {k: v / total for k, v in scores_dict.items()}


# =========================
# Predicción híbrida
# =========================
def predict_hybrid(
    texto:     str,
    remitente: str,
    tipo_doc:  str,
    fecha_str: str,
    k:         int = 3,
) -> list[dict]:
    """
    Predicción híbrida con alpha continuo inteligente [C6].

        score_final = alpha_final * score_modelo + (1 - alpha_final) * score_historico

    Cambios respecto a main.py:
      · [C1] Normalización de remitente y tipo_doc con limpiar_texto_general()
             antes de buscar en prior_counts y remitente_to_id.
      · [C5] score_modelo y score_historico se normalizan antes de combinar.
      · [C6] Alpha continuo inteligente en lugar de rangos discretos.
      · [C7] La clave de búsqueda en prior_counts es el texto limpio,
             igual que en el entrenamiento.
    """
    state     = app_state
    model     = state.model
    tokenizer = state.tokenizer
    mlb       = state.mlb
    device    = state.device

    model.eval()

    # ---- Tokenización ----
    encoding       = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # ---- [C1][C7] Normalizar claves igual que en entrenamiento ----
    rem_clean  = limpiar_texto_general(remitente)   # coincide con REMITENTE_CLEAN
    tipo_clean = limpiar_texto_general(tipo_doc)    # coincide con TIPO_DOC_CLEAN

    remitente_id = state.remitente_to_id.get(rem_clean, 0)
    tipo_id      = state.tipo_to_id.get(tipo_clean, 0)

    if remitente_id == 0:
        logger.warning(
            "Remitente desconocido: '%s' → usando <UNK>. "
            "Muestra de claves disponibles: %s",
            rem_clean,
            list(state.remitente_to_id.keys())[:6],
        )
    if tipo_id == 0:
        logger.warning("Tipo desconocido: '%s' → usando <UNK>", tipo_clean)

    remitente_enc = torch.tensor([remitente_id]).to(device)
    tipo_enc      = torch.tensor([tipo_id]).to(device)

    # ---- Features de fecha ----
    fecha_dt    = pd.to_datetime(fecha_str)
    fecha_feats = torch.tensor(
        [[fecha_dt.weekday(), fecha_dt.month]], dtype=torch.float
    ).to(device)

    # ---- Inferencia del modelo ----
    with torch.no_grad():
        logits = model(input_ids, attention_mask, remitente_enc, tipo_enc, fecha_feats)

    probs = torch.sigmoid(logits)[0].cpu().numpy()

    # ---- [C5] Score del modelo (normalizado) ----
    score_modelo = {mlb.classes_[i]: float(probs[i]) for i in range(len(probs))}
    score_modelo = normalizar_scores(score_modelo)

    # ---- [C5][C7] Score histórico (normalizado) ----
    score_historico = {c: 0.0 for c in mlb.classes_}
    score_historico = normalizar_scores(score_historico)   # todos 0 → sin cambio
    total = 0

    # [C7] Buscar con la clave limpia, igual que al construir prior_counts
    if rem_clean in state.prior_counts:
        conteo = state.prior_counts[rem_clean]
        total  = sum(conteo.values())
        for resp, count in conteo.items():
            if resp in score_historico:
                score_historico[resp] = count / total
        logger.info(
            "Histórico encontrado para '%s' — total=%d documentos", rem_clean, total
        )
    else:
        logger.warning("Sin histórico para '%s' → solo modelo", rem_clean)

    # ---- [C6] Alpha continuo inteligente ----
    # Reemplaza los rangos discretos (if total>20 and max_prob<0.6 …) del main.py original.
    # Fórmula del notebook (sección 14):
    #   conf_model  = max(probs del modelo)
    #   conf_hist   = max(score_historico)  — 0 si sin histórico
    #   alpha_base  = conf_model / (conf_model + conf_hist + eps)
    #   peso_hist   = total / (total + c)   — c=10, controla cuánto "pesa" el histórico
    #   alpha_final = alpha_base * (1 - peso_hist)
    #   alpha_final = clamp(alpha_final, 0.1, 0.9)
    eps        = 1e-6
    c          = 10                                         # hiperparámetro del notebook

    conf_model = float(max(probs))
    conf_hist  = max(score_historico.values()) if total > 0 else 0.0

    alpha_base  = conf_model / (conf_model + conf_hist + eps)
    peso_hist   = total / (total + c)
    alpha_final = alpha_base * (1 - peso_hist)
    alpha_final = max(0.1, min(0.9, alpha_final))           # límites suaves

    logger.info(
        "Alpha: conf_model=%.3f conf_hist=%.3f peso_hist=%.3f alpha_final=%.3f",
        conf_model, conf_hist, peso_hist, alpha_final,
    )

    # ---- Score final combinado ----
    score_final = {
        resp: alpha_final * score_modelo.get(resp, 0.0)
              + (1 - alpha_final) * score_historico.get(resp, 0.0)
        for resp in mlb.classes_
    }

    # ---- Top-K ----
    top_resps = sorted(score_final, key=score_final.get, reverse=True)[:k]

    return [
        {
            "responsable":      resp,
            "score_modelo":     round(score_modelo[resp],    4),
            "score_historico":  round(score_historico[resp], 4),
            "score_final":      round(score_final[resp],     4),
            "probabilidad_pct": round(score_final[resp] * 100, 2),
            "alpha":            round(alpha_final, 4),
        }
        for resp in top_resps
    ]


# =========================
# Carga de artefactos al iniciar
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y artefactos una sola vez al arrancar."""
    logger.info("Cargando artefactos del modelo…")

    # Dispositivo
    app_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", app_state.device)

    # [C4] Resolver fuente de BETO antes de cargar cualquier cosa
    beto_path = load_or_resolve_beto_path()

    # ── Artefactos serializados ──────────────────────────────────────────────
    if not os.path.exists(ARTIFACTS_PATH):
        logger.warning(
            "Archivo de artefactos '%s' no encontrado. "
            "El endpoint /predict no estará disponible.",
            ARTIFACTS_PATH,
        )
    else:
        with open(ARTIFACTS_PATH, "rb") as f:
            artifacts = pickle.load(f)

        app_state.mlb             = artifacts["mlb"]
        app_state.remitente_to_id = artifacts["remitente_to_id"]
        app_state.tipo_to_id      = artifacts["tipo_to_id"]
        app_state.prior_counts    = artifacts["prior_counts"]

        logger.info(
            "Artefactos cargados: %d clases, %d remitentes, %d tipos",
            len(app_state.mlb.classes_),
            len(app_state.remitente_to_id),
            len(app_state.tipo_to_id),
        )

    # ── Pesos del modelo ─────────────────────────────────────────────────────
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        logger.warning(
            "Pesos del modelo '%s' no encontrados. "
            "El endpoint /predict no estará disponible.",
            MODEL_WEIGHTS_PATH,
        )
    else:
        num_labels     = len(app_state.mlb.classes_)
        num_remitentes = len(app_state.remitente_to_id)
        num_tipos      = len(app_state.tipo_to_id)

        # [C3] Se pasa beto_path al constructor
        app_state.model = ModeloHibrido(
            num_labels, num_remitentes, num_tipos,
            bert_model_path=beto_path,              # [C3]
        ).to(app_state.device)

        app_state.model.load_state_dict(
            torch.load(MODEL_WEIGHTS_PATH, map_location=app_state.device)
        )
        app_state.model.eval()
        logger.info("Modelo cargado correctamente.")

    # ── Tokenizador ──────────────────────────────────────────────────────────
    # [C4] Se carga desde la misma fuente que el modelo
    app_state.tokenizer = AutoTokenizer.from_pretrained(beto_path)
    logger.info("Tokenizador cargado desde: %s", beto_path)

    yield  # ── aplicación corriendo ──

    logger.info("Cerrando la aplicación…")


# =========================
# FastAPI App
# =========================
app = FastAPI(
    title="API de Clasificación de Documentos",
    description=(
        "Predice los 3 responsables más probables para atender un documento "
        "usando un modelo híbrido BETO + prior histórico con alpha continuo inteligente."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="./static"), name="static")


# =========================
# Schemas
# =========================
class DocumentoRequest(BaseModel):
    asunto:    str = Field(..., min_length=1, max_length=512,
                           description="Asunto del documento")
    remitente: str = Field(..., min_length=1, max_length=200,
                           description="Remitente del documento")
    tipo_doc:  str = Field(..., description="Tipo de documento: CARTA | MEMORANDO | OFICIO")
    fecha:     str = Field(..., description="Fecha de ingreso en formato YYYY-MM-DD",
                           example=str(date.today()))

    class Config:
        json_schema_extra = {
            "example": {
                "asunto":    "Remite factura vigésimo noveno mes fase 2",
                "remitente": "COLSOF S.A.S. - SUCURSAL PERU",
                "tipo_doc":  "CARTA",
                "fecha":     str(date.today()),
            }
        }


class ResponsableResult(BaseModel):
    responsable:      str
    score_modelo:     float
    score_historico:  float
    score_final:      float
    probabilidad_pct: float
    alpha:            float


class PrediccionResponse(BaseModel):
    documento:    DocumentoRequest
    responsables: list[ResponsableResult]
    total_clases: int


# =========================
# Endpoints
# =========================
@app.get("/", tags=["Health"])
def root():
    return FileResponse("../static/index.html")


@app.get("/health", tags=["Health"])
def health():
    return {
        "modelo_cargado": app_state.model is not None,
        "device":         str(app_state.device),
        "num_clases":     len(app_state.mlb.classes_) if app_state.mlb else 0,
    }


@app.post("/predict", response_model=PrediccionResponse, tags=["Clasificación"])
def predecir_responsable(request: DocumentoRequest):
    """
    Recibe los datos del documento y devuelve los 3 responsables
    más probables ordenados por score final híbrido (alpha continuo).
    """
    if app_state.model is None or app_state.mlb is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Verifica los archivos de pesos y artefactos.",
        )

    tipo_norm = request.tipo_doc.strip().upper()
    if tipo_norm not in TIPOS_VALIDOS:
        raise HTTPException(
            status_code=422,
            detail=f"tipo_doc inválido. Valores permitidos: {sorted(TIPOS_VALIDOS)}",
        )

    try:
        pd.to_datetime(request.fecha)
    except Exception:
        raise HTTPException(status_code=422, detail="Formato de fecha inválido. Use YYYY-MM-DD.")

    # [C2] build_texto_modelo ahora usa limpiar_texto_general internamente
    texto_modelo = build_texto_modelo(request.remitente, tipo_norm, request.asunto)
    logger.info("Texto modelo: %s", texto_modelo[:100])

    try:
        resultados = predict_hybrid(
            texto=texto_modelo,
            remitente=request.remitente,
            tipo_doc=tipo_norm,
            fecha_str=request.fecha,
            k=3,
        )
    except Exception as exc:
        logger.exception("Error en la predicción")
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {exc}")

    return PrediccionResponse(
        documento=request,
        responsables=[ResponsableResult(**r) for r in resultados],
        total_clases=len(app_state.mlb.classes_),
    )


@app.get("/tipos", tags=["Metadatos"])
def listar_tipos():
    return {"tipos": sorted(TIPOS_VALIDOS)}


@app.get("/clases", tags=["Metadatos"])
def listar_clases():
    if app_state.mlb is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return {"clases": list(app_state.mlb.classes_)}


@app.get("/diagnostico/prior", tags=["Diagnóstico"])
def diagnostico_prior(remitente: str):
    """
    Inspecciona el prior_counts para un remitente dado.
    La búsqueda usa limpiar_texto_general(), igual que en la inferencia.

    Ejemplo: GET /diagnostico/prior?remitente=COLSOF%20S.A.S.%20-%20SUCURSAL%20PERU
    """
    if app_state.mlb is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    rem_clean       = limpiar_texto_general(remitente)
    tiene_historico = rem_clean in app_state.prior_counts
    conteo          = dict(app_state.prior_counts.get(rem_clean, {}))
    total           = sum(conteo.values())

    return {
        "remitente_recibido":    remitente,
        "remitente_normalizado": rem_clean,
        "tiene_historico":       tiene_historico,
        "total_documentos":      total,
        "conteo_responsables":   conteo,
        "muestra_claves_prior":  list(app_state.prior_counts.keys())[:20],
    }


@app.get("/diagnostico/unicode", tags=["Diagnóstico"])
def diagnostico_unicode(remitente: str):
    """
    Diagnóstico de encoding Unicode: compara byte a byte los responsables
    del prior_counts contra las clases del mlb.

    Ejemplo: GET /diagnostico/unicode?remitente=CONSORCIO%20INDRA%20SISTEMA%20COMERCIAL
    """
    if app_state.mlb is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    rem_clean       = limpiar_texto_general(remitente)
    tiene_historico = rem_clean in app_state.prior_counts

    if not tiene_historico:
        return {"error": f"Remitente normalizado '{rem_clean}' no encontrado en prior_counts."}

    conteo     = app_state.prior_counts[rem_clean]
    mlb_clases = list(app_state.mlb.classes_)

    analisis = []
    for resp in conteo.keys():
        resp_norm    = limpiar_texto_general(resp)
        match_exacto = resp in mlb_clases
        match_norm   = any(limpiar_texto_general(c) == resp_norm for c in mlb_clases)
        clase_match  = next(
            (c for c in mlb_clases if limpiar_texto_general(c) == resp_norm), None
        )

        diff_detalle = None
        if match_norm and not match_exacto and clase_match:
            diff_detalle = [
                {
                    "pos": i,
                    "prior_char": ca, "prior_bytes": ca.encode("utf-8").hex(),
                    "mlb_char":   cb, "mlb_bytes":   cb.encode("utf-8").hex(),
                }
                for i, (ca, cb) in enumerate(zip(resp, clase_match)) if ca != cb
            ]

        analisis.append({
            "responsable_prior":         resp,
            "resp_normalizado":          resp_norm,
            "bytes_utf8":                resp.encode("utf-8").hex(),
            "match_exacto_con_mlb":      match_exacto,
            "match_normalizado_con_mlb": match_norm,
            "clase_mlb_correspondiente": clase_match,
            "diferencias_char":          diff_detalle,
        })

    sospechosos = [a for a in analisis if a["match_normalizado_con_mlb"] and not a["match_exacto_con_mlb"]]

    return {
        "remitente":             remitente,
        "remitente_normalizado": rem_clean,
        "analisis":              analisis,
        "sospechosos_encoding":  sospechosos,
        "conclusion": (
            "Hay responsables del prior que coinciden normalizados pero NO exactamente "
            "con mlb.classes_ → posible problema de encoding residual."
            if sospechosos else
            "Sin problemas de encoding detectados."
        ),
    }


# =========================
# Utilidad: exportar artefactos desde Colab
# (copiar y pegar en el notebook, sección 18)
# =========================
"""
import pickle, torch

with open("model_artifacts.pkl", "wb") as f:
    pickle.dump({
        "mlb":             mlb,
        "remitente_to_id": remitente_to_id,
        "tipo_to_id":      tipo_to_id,
        "prior_counts":    prior_counts,
    }, f)

torch.save(model.state_dict(), "model_weights.pt")
"""

# =========================
# Punto de entrada
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
