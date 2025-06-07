import os

import torch
import torchvision
import numpy
import faiss
import PIL
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
resnet_model = resnet_model.to(device).eval()

efficientnet_model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
efficientnet_model.classifier = torch.nn.Identity()
efficientnet_model = efficientnet_model.to(device).eval()

def extract_features(model, transform, image_path_or_pil):
    if isinstance(image_path_or_pil, str):
        image = PIL.Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
        embedding = features.squeeze().detach().cpu().numpy().astype("float32")
        return embedding

def process_and_save(model, name, image_paths, save_dir="./data"):
    embeddings, paths = [], []
    for path in tqdm.tqdm(image_paths, desc=f"Extracting [{name}] embeddings"):
        try:
            emb = extract_features(model, transform, path)
            embeddings.append(emb)
            paths.append(path)
        except Exception as e:
            print(f"[{name}] Error with {path}: {e}")
    embeddings = numpy.array(embeddings)
    paths = numpy.array(paths)
    numpy.save(f"{save_dir}/{name}_embs.npy", embeddings)
    numpy.save(f"{save_dir}/{name}_paths.npy", paths)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f"{save_dir}/{name}.index")

def get_top5(image, app_state):
    resnet_model = app_state.resnet_model
    efficientnet_model = app_state.efficientnet_model
    transform = app_state.transform
    resnet_index = app_state.resnet_index
    efficientnet_index = app_state.efficientnet_index
    resnet_image_paths = app_state.resnet_image_paths
    efficientnet_image_paths = app_state.efficientnet_image_paths

    resnet_emb = extract_features(resnet_model, transform, image)
    effnet_emb = extract_features(efficientnet_model, transform, image)

    resnet_dist, resnet_idx = resnet_index.search(numpy.expand_dims(resnet_emb, 0), 5)
    effnet_dist, effnet_idx = efficientnet_index.search(numpy.expand_dims(effnet_emb, 0), 5)

    resnet_results = {
        str(resnet_image_paths[i]): float(resnet_dist[0][j])
        for j, i in enumerate(resnet_idx[0])
    }
    effnet_results = {
        str(efficientnet_image_paths[i]): float(effnet_dist[0][j])
        for j, i in enumerate(effnet_idx[0])
    }

    return resnet_results, effnet_results
