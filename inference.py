from ray import serve
from fastapi import FastAPI
from predictor import Predictor
from omegaconf import OmegaConf
from starlette.requests import Request
from typing import Dict
import os

app = FastAPI()

@serve.deployment(route_prefix="/inference")
class MyModel:
    def __init__(self) -> None:
        cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
        self.predictor = Predictor(cfg)
    
    async def __call__(self, request: Request) -> Dict:

        try:
            data = await request.json()
        except RuntimeError:
            data = "Receive channel not available"

        instances = data["instances"]

        results = self.predictor.predict(instances)

        return {"results": results.tolist()}

model = MyModel.bind()