# import required libraries
import h2o
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, StrictFloat, confloat 
from typing import Dict
import pathlib


# FastAPI instance
app = FastAPI()

# Request validation and response management
class RequestItem(BaseModel):
    requestID: str
    sepal_length_cm: confloat(strict=False, ge=0.5, le=10.0)
    sepal_width_cm: confloat(strict=False, ge=0.5, le=10.0)
    petal_length_cm: confloat(strict=False, ge=0.5, le=10.0)
    petal_width_cm: confloat(strict=False, ge=0.5, le=10.0)

class ResponseItem(BaseModel):
    requestID: str
    status: str
    prediction: str
    setosa_proba: float
    versicolor_proba: float
    virginica_proba: float

# start up an H2O instance and import model
# It is extremely important to initiate h2o outside of Python using Java. Extreme performance issue when h2o instantiate from Python
h2o.init()

# point model & file locations
model_path = pathlib.Path(r'modelfiles/irisgbm_20230123.zip')

# upload H2O model files into the H2O instance 
imported_model = h2o.upload_mojo(str(model_path))

# predict results
@app.post("/prediction", response_model=ResponseItem)
def model_serve(request: RequestItem):
    input_hdf = h2o.H2OFrame(dict(request))
    p_prediction, setosa_proba, versicolor_proba, virginica_proba = h2o.as_list(imported_model.predict(input_hdf), use_pandas=False)[1]
    return {
            "requestID": request.requestID, 
            "status": "Success",
            "prediction": p_prediction,
            "setosa_proba": setosa_proba,
            "versicolor_proba": versicolor_proba,
            "virginica_proba": virginica_proba
    }

# health check
@app.get("/healthz")
def health_check():
    return {"api_health": True}

# API Schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(title="FastAPI", version="0.1.0", routes=app.routes)
    openapi_schema["x-googel-backend"] = {
        "address": "${CLOUD_RUN_URL}",
        "deadline": "${TIMEOUT}"
    }
    openapi_schema["paths"]["/prediction"]["options"] = {
        "operationID": "corsHellow",
        "response": {"200": {"description": "Successful response"}},
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# to run locally: run "uvicorn score_API:app --reload --port 8000" in shell