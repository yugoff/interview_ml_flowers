import torch
import torchvision
import numpy
import faiss
import uvicorn
import fastapi
import PIL
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

resnet_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
resnet_model.fc = torch.nn.Identity()
resnet_model = resnet_model.to(device)
resnet_model.eval()

efficientnet_model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
efficientnet_model.classifier = torch.nn.Identity()
efficientnet_model = efficientnet_model.to(device)
efficientnet_model.eval()

resnet_embeddings = numpy.load("./data/resnet_embs.npy")
resnet_image_paths = numpy.load("./data/resnet_paths.npy")
resnet_index = faiss.read_index("./data/resnet.index")

efficientnet_embeddings = numpy.load("./data/efficientnet_embs.npy")
efficientnet_image_paths = numpy.load("./data/efficient_path.npy")
efficientnet_index = faiss.read_index("./data/efficientnet.index")

def extract_features(model, transform, image: PIL.Image.Image) -> numpy.ndarray:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
        embedding = features.squeeze().detach().cpu().numpy().astype("float32")
        return embedding

def get_top5(image: PIL.Image.Image):
    resnet_embedding = extract_features(resnet_model, transform, image)
    efficientnet_embedding = extract_features(efficientnet_model, transform, image)
    resnet_distances, resnet_indices = resnet_index.search(numpy.expand_dims(resnet_embedding, axis=0), 5)
    efficientnet_distances, efficientnet_indices = efficientnet_index.search(numpy.expand_dims(efficientnet_embedding, axis=0), 5)
    resnet_results = {
        str(resnet_image_paths[i]): float(resnet_distances[0][j])
        for j, i in enumerate(resnet_indices[0])
    }
    efficientnet_results = {
        str(efficientnet_image_paths[i]): float(efficientnet_distances[0][j])
        for j, i in enumerate(efficientnet_indices[0])
    }
    return resnet_results, efficientnet_results

app = fastapi.FastAPI()

@app.post("/search")
async def search(file: fastapi.UploadFile = fastapi.File(...)):
    contents = await file.read()
    try:
        image = PIL.Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return fastapi.responses.JSONResponse(status_code=400, content={"error": "Invalid image format"})

    results = get_top5(image)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
