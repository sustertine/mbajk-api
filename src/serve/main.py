# -*- coding: utf-8 -*-

from fastapi import FastAPI
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from src.models.input_model import InputModel
from src.models.input_transformer import InputTransformer
from src.models.predictor import Predictor
from src.models.station_service import get_stations_locations

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

input_transformer = InputTransformer()
model = Predictor()


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.post("/api/mbajk/predict")
def predict(data: InputModel):
    transformed_input = input_transformer.transform(data)
    prediction = model.predict_station(data.station_name, transformed_input).tolist()
    return {
        "prediction": prediction
    }


@app.get("/api/mbajk/stations")
def get_stations():
    return list(model.model_map.keys())


@app.get("/api/mbajk/stations/info")
def get_stations_info():
    return get_stations_locations()

@app.get("/api/mbajk/stations/model_map")
def get_model_map():
    return model.return_model_map()