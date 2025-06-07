import os

import fastapi
import contextlib
import uvicorn
import numpy
import faiss

from models import process_and_save, resnet_model, efficientnet_model, transform
from api.routes import router


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    image_dir = "./flowers/train"
    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    os.makedirs("./data", exist_ok=True)

    if not os.path.exists("./data/resnet_embs.npy"):
        print("ResNet эмбеддинги не найдены — создаём...")
        process_and_save(resnet_model, "resnet", image_paths)
    else:
        print("ResNet эмбеддинги уже существуют")

    if not os.path.exists("./data/efficientnet_embs.npy"):
        print("EfficientNet эмбеддинги не найдены — создаём...")
        process_and_save(efficientnet_model, "efficientnet", image_paths)
    else:
        print("EfficientNet эмбеддинги уже существуют")

    app.state.resnet_model = resnet_model
    app.state.efficientnet_model = efficientnet_model
    app.state.transform = transform

    app.state.resnet_embeddings = numpy.load("./data/resnet_embs.npy")
    app.state.resnet_image_paths = numpy.load("./data/resnet_paths.npy")
    app.state.resnet_index = faiss.read_index("./data/resnet.index")

    app.state.efficientnet_embeddings = numpy.load("./data/efficientnet_embs.npy")
    app.state.efficientnet_image_paths = numpy.load("./data/efficientnet_paths.npy")
    app.state.efficientnet_index = faiss.read_index("./data/efficientnet.index")

    yield

app = fastapi.FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
