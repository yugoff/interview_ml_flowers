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

transform_effnet = torchvision.transforms.Compose([
    torchvision.transforms.Resize((300, 300)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])

effnet_model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0, global_pool='avg')
effnet_model = effnet_model.to(device)
effnet_model.eval()

effnet_embeddings = numpy.load("./data_2/effnet_embs.npy")
image_paths = numpy.load("./data_2/paths.npy")
effnet_index = faiss.read_index("./data_2/effnet.index")

def extract_features(model, transform, image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
        embedding = features.squeeze().detach().cpu().numpy().astype("float32")
        return embedding

def get_top5(image: Image.Image):
    embedding = extract_features(effnet_model, transform_effnet, image)
    distances, indices = effnet_index.search(numpy.expand_dims(embedding, axis=0), 5)
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
