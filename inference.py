from ray import serve
from fastapi import FastAPI
from predictor import Predictor
from omegaconf import OmegaConf
from starlette.requests import Request
from typing import Dict

app = FastAPI()

cfg = OmegaConf.load("./config.yaml")

@serve.deployment(route_prefix="{}".format(cfg.route))
class MyModel:
    def __init__(self) -> None:
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