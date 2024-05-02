from pydantic import BaseModel


class InputModel(BaseModel):
    temperature: float = 25.1
    relative_humidity: float = 45
    dew_point: float = 12.4
    apparent_temperature: float = 24.7
    precipitation_probability: float = 0.0
    rain: float = 0.0
    surface_pressure: float = 984.3
    bike_stands: int = 22
    available_bike_stands: int = 8
