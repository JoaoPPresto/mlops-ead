import os
import logging
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import mlflow

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Inicialização do app
app = FastAPI(
    title="Fetal Health API",
    openapi_tags=[
        {"name": "Health", "description": "Get API health"},
        {"name": "Prediction", "description": "Model prediction"}
    ]
)

# Modelo de entrada com validação
class FetalHealthData(BaseModel):
    accelerations: float = Field(..., ge=0)
    fetal_movement: float = Field(..., ge=0)
    uterine_contractions: float = Field(..., ge=0)
    severe_decelerations: float = Field(..., ge=0)

# Tratamento de exceções genéricas
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erro inesperado: {exc}")
    return JSONResponse(status_code=500, content={"error": "Erro interno no servidor"})

# Carregamento do modelo
def load_model():
    logger.info("Carregando modelo...")
    tracking_uri = 'https://dagshub.com/renansantosmendes/puc_lectures_mlops.mlflow'
    # os.getenv("MLFLOW_TRACKING_URI")
    username = 'renansantosmendes'
    # os.getenv("MLFLOW_TRACKING_USERNAME")
    password = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
    # os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not all([tracking_uri, username, password]):
        raise EnvironmentError("Credenciais do MLflow não configuradas corretamente.")

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    model_info = client.get_registered_model("fetal_health")
    run_id = model_info.latest_versions[-1].run_id
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    logger.info("Modelo carregado com sucesso.")
    return model

# Evento de inicialização
@app.on_event("startup")
def startup_event():
    app.state.model = load_model()

# Endpoint de saúde
@app.get("/", tags=["Health"])
def api_health():
    return {"status": "healthy"}

# Endpoint de predição
@app.post("/predict", tags=["Prediction"])
def predict(request: FetalHealthData):
    model = app.state.model
    input_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations
    ]).reshape(1, -1)

    logger.info(f"Dados recebidos: {input_data}")
    prediction = model.predict(input_data)
    predicted_class = int(np.argmax(prediction[0]))
    classes = ["Normal", "Suspeito", "Patológico"]

    return {
        "prediction": classes[predicted_class],
        "probabilities": prediction[0].tolist()
    }