from fastapi import FastAPI
from predictor import Predictor
from omegaconf import OmegaConf
from starlette.requests import Request
from typing import Dict
import os
import ray
from ray import serve
from ray.serve.handle import RayServeDeploymentHandle

app = FastAPI()

@serve.deployment
class GTPImmobilized():
    def __init__(self) -> None:
        cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
        self.predictor = Predictor(cfg, model_path=os.path.join(os.path.dirname(__file__), cfg.checkpoints.gtp_immobilized.model_path ))
    
    async def predict(self, instances: Dict) -> Dict:

        results = self.predictor.predict(instances)

        return {"results": results.tolist()}


@serve.deployment
class GTPSoluble():
    def __init__(self) -> None:
        cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
        self.predictor = Predictor(cfg, model_path=os.path.join(os.path.dirname(__file__), cfg.checkpoints.gtp_soluble.model_path ))
    
    async def predict(self, instances: Dict) -> Dict:

        results = self.predictor.predict(instances)

        return {"results": results.tolist()}


@serve.deployment
class GDPImmobilized():
    def __init__(self) -> None:
        cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
        self.predictor = Predictor(cfg, model_path=os.path.join(os.path.dirname(__file__), cfg.checkpoints.gdp_immobilized.model_path ))
    
    async def predict(self, instances: Dict) -> Dict:

        results = self.predictor.predict(instances)

        return {"results": results.tolist()}


@serve.deployment
class GDPSoluble():
    def __init__(self) -> None:
        cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
        self.predictor = Predictor(cfg, model_path=os.path.join(os.path.dirname(__file__), cfg.checkpoints.gdp_soluble.model_path ))
    
    async def predict(self, instances: Dict) -> Dict:
    
        results = self.predictor.predict(instances)

        return {"results": results.tolist()}


@serve.deployment
class KrasG12D():
    def __init__(self) -> None:
        cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
        self.predictor = Predictor(cfg, model_path=os.path.join(os.path.dirname(__file__), cfg.checkpoints.kras_g12d.model_path ))
    
    async def predict(self, instances: Dict) -> Dict:

        results = self.predictor.predict(instances)

        return {"results": results.tolist()}


@serve.deployment
class KrasG12V():
    def __init__(self) -> None:
        cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
        self.predictor = Predictor(cfg, model_path=os.path.join(os.path.dirname(__file__), cfg.checkpoints.kras_g12v.model_path ))
    
    async def predict(self, instances: Dict) -> Dict:
    
        results = self.predictor.predict(instances)

        return {"results": results.tolist()}



@serve.deployment(route_prefix="/mrs/v1/predict")
class MrsV1Graph():
    def __init__(self, 
        gtp_immobilized_model: RayServeDeploymentHandle,
        gdp_immobilized_model: RayServeDeploymentHandle,
        gtp_soluble_model: RayServeDeploymentHandle,
        gdp_soluble_model: RayServeDeploymentHandle,
        kras_g12v_model: RayServeDeploymentHandle,
        kras_g12d_model: RayServeDeploymentHandle
    ) -> None:

        self.directory = {
            "gtp_immobilized_model": gtp_immobilized_model,
            "gdp_immobilized_model": gdp_immobilized_model,
            "gtp_soluble_model": gtp_soluble_model,
            "gdp_soluble_model": gdp_soluble_model,
            "kras_g12v_model": kras_g12v_model,
            "kras_g12d_model": kras_g12d_model
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


gtp_immobilized_model = GTPImmobilized.bind()
gdp_immobilized_model = GDPImmobilized.bind()
gtp_soluble_model = GTPSoluble.bind()
gdp_soluble_model  = GDPSoluble.bind()
kras_g12d_model = KrasG12D.bind()
kras_g12v_model  = KrasG12V.bind()


model = MrsV1Graph.bind(gtp_immobilized_model, gdp_immobilized_model, gtp_soluble_model, gdp_soluble_model, kras_g12v_model, kras_g12d_model)
    




