import os
import torch
import torchvision
import timm
import numpy
import faiss
import PIL.Image
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_effnet = torchvision.transforms.Compose([
    torchvision.transforms.Resize((300, 300)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])

effnet_model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0, global_pool='avg')
effnet_model = effnet_model.to(device)
effnet_model.eval()

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
        emb = extract_features(effnet_model, transform_effnet, path)
        embeddings.append(emb)
        paths.append(path)
    except Exception as e:
        print(f"Error with {path}: {e}")

os.makedirs("./data_2", exist_ok=True)
embeddings = numpy.array(embeddings)
paths = numpy.array(paths)

numpy.save("./data_2/effnet_embs.npy", embeddings)
numpy.save("./data_2/paths.npy", paths)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "./data_2/effnet.index")
