from fastapi import FastAPI
from starlette.responses import RedirectResponse

from src.models.input_model import InputModel
from src.models.input_transformer import InputTransformer
from src.models.predictor import Predictor

app = FastAPI()

input_transformer = InputTransformer()
model = Predictor()


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.post("/api/mbajk/predict")
def predict(data: InputModel):
    transformed_input = input_transformer.transform(data)
    prediction = model.predict(transformed_input).tolist()
    return {
        "prediction": prediction
    }
