from fastapi import FastAPI
from service.api.api import main_router
import onnxruntime as rt

app = FastAPI(project_name="Emotions Detection")
app.include_router(main_router)

providers=['CUDAExecutionProvider']
m_q = rt.InferenceSession("service/vit_keras.onnx", providers=providers)

@app.get("/")
async def root():
    return {"hello": "world"}
