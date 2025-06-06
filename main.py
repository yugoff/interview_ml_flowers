import os
import torch
import torchvision
import timm
import numpy
import faiss
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_resnet = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])

resnet_model = torchvision.models.resnet50(pretrained=True)
resnet_model.fc = torch.nn.Identity()
resnet_model = resnet_model.to(device)
resnet_model.eval()

resnet_embeddings = numpy.load("./data/resnet_embs.npy")
image_paths = numpy.load("./data/paths.npy")
resnet_index = faiss.read_index("./data/resnet.index")

def extract_features(model, transform, image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
        embedding = features.squeeze().detach().cpu().numpy().astype("float32")
        return embedding

def get_top5(image: Image.Image):
    embedding = extract_features(resnet_model, transform_resnet, image)
    distances, indices = resnet_index.search(numpy.expand_dims(embedding, axis=0), 5)
    results = {
        str(image_paths[i]): float(distances[0][j])
        for j, i in enumerate(indices[0])
    }
    return results

app = FastAPI()

@app.post("/search")
async def search(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    results = get_top5(image)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
