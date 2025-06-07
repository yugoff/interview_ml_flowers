import os
import torch
import torchvision
import numpy
import faiss
import PIL.Image
import tqdm

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

def process_and_save(model, name, save_dir="./data"):
    embeddings = []
    paths = []

    for path in tqdm.tqdm(image_paths, desc=f"Extracting [{name}] embeddings"):
        try:
            emb = extract_features(model, transform, path)
            embeddings.append(emb)
            paths.append(path)
        except Exception as e:
            print(f"[{name}] Error with {path}: {e}")

    os.makedirs(save_dir, exist_ok=True)
    embeddings = numpy.array(embeddings)
    paths = numpy.array(paths)

    numpy.save(f"{save_dir}/{name}_embs.npy", embeddings)
    numpy.save(f"{save_dir}/{name}_paths.npy", paths)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f"{save_dir}/{name}.index")

process_and_save(resnet_model, name="resnet")
process_and_save(efficientnet_model, name="efficientnet")
