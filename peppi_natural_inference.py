from ray import serve
from fastapi import FastAPI
from predictor import Predictor
from omegaconf import OmegaConf
from starlette.requests import Request
from typing import Dict
import os
from ray.serve.handle import RayServeDeploymentHandle
import ray

app = FastAPI()


@serve.deployment
class PeppiNatEpoch7():
    def __init__(self) -> None:
        cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
        self.predictor = Predictor(cfg, model_path=os.path.join(os.path.dirname(__file__), cfg.checkpoints.epoch_7.model_path ))
    
    async def predict(self, instances: Dict) -> Dict:

        results = self.predictor.predict(instances)

        return {"results": results.tolist()}


@serve.deployment
class PeppiNatEpoch20():
    def __init__(self) -> None:
        cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
        self.predictor = Predictor(cfg, model_path=os.path.join(os.path.dirname(__file__), cfg.checkpoints.epoch_20.model_path ))
    
    async def predict(self, instances: Dict) -> Dict:
    
        results = self.predictor.predict(instances)

        return {"results": results.tolist()}


@serve.deployment(route_prefix="/peppi/natural/v1/predict")
class PepPIGraph():
    def __init__(self, 
        peppi_nat_epoch7_model: RayServeDeploymentHandle,
        peppi_nat_epoch20_model: RayServeDeploymentHandle,
    ) -> None:

        self.directory = {
            "peppi_nat_epoch7_model": peppi_nat_epoch7_model,
            "peppi_nat_epoch20_model": peppi_nat_epoch20_model,
        }
        
    async def __call__(self, request: Request) -> Dict:

        try:
            data = await request.json()
        except RuntimeError:
            data = "Receive channel not available"

        instances = data["instances"]
        model_name = data["model_name"]

        ref: ray.ObjectRef = await self.directory[model_name].predict.remote(instances)

        if isinstance(ref, ray.ObjectRef):
            results = await ref
        else:
            results = ref
        
        return results


peppi_nat_epoch7_model = PeppiNatEpoch7.bind()
peppi_nat_epoch20_model = PeppiNatEpoch20.bind()

model = PepPIGraph.bind(peppi_nat_epoch7_model, peppi_nat_epoch20_model)