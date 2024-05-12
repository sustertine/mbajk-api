# -*- coding: utf-8 -*-
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pymongo import MongoClient
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from src.models.api_logger import ApiLogger, ApiLogEntry
from src.models.current_data_service import CurrentDataService
from src.models.input_model import StationInputModel
from src.models.input_transformer import InputTransformer
from src.models.predictor import Predictor
from src.models.station_service import get_stations_locations
from sklearn import set_config

load_dotenv()
set_config(transform_output="pandas")

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
current_data_service = CurrentDataService()
api_logger = ApiLogger()

@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.post("/api/mbajk/predict")
def predict(data: StationInputModel):
    station_name = data.station_name
    data = current_data_service.get_current_data(station_name)

    log_entry = ApiLogEntry.from_df(data)
    log_entry.station_name = station_name

    prediction = model.predict_station(station_name, data).tolist()
    log_entry.predictions = prediction
    api_logger.log(log_entry)
    return {
        "prediction": prediction
    }

@app.post("/api/mbajk/stations/available_bike_stands")
def get_available_bike_stands(data: StationInputModel):
    return {
        'available_bike_stands': int(current_data_service.get_available_bike_stands(data.station_name))
    }


@app.get("/api/mbajk/stations")
def get_stations():
    return list(model.model_map.keys())


@app.get("/api/mbajk/stations/info")
def get_stations_info():
    return get_stations_locations()