import os
import torch
import torchvision
import timm
import numpy
import faiss
import PIL.Image
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_resnet = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])

resnet_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
resnet_model.fc = torch.nn.Identity()
resnet_model = resnet_model.to(device)
resnet_model.eval()

def extract_features(model, transform, image_path):
    try:
        image = PIL.Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"[Open error] {image_path}: {e}")

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
        embedding = features.squeeze().detach().cpu().numpy().astype("float32")
        return embedding

image_dir = "./flowers/train"
image_paths = [
    os.path.join(image_dir, f) for f in os.listdir(image_dir)
    if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
]

embeddings = []
paths = []

for path in tqdm.tqdm(image_paths, desc="Extracting embeddings"):
    try:
        emb = extract_features(resnet_model, transform_resnet, path)
        embeddings.append(emb)
        paths.append(path)
    except Exception as e:
        print(f"Error with {path}: {e}")

os.makedirs("./data", exist_ok=True)
embeddings = numpy.array(embeddings)
paths = numpy.array(paths)

numpy.save("./data/resnet_embs.npy", embeddings)
numpy.save("./data/paths.npy", paths)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "./data/resnet.index")
