from fastapi import FastAPI, File, UploadFile
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import io
import os
import requests
import json
import hashlib
import torch
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

load_dotenv()
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

app = FastAPI()

# Cache directories
CACHE_DIR = "./image_cache"
EMBEDDING_CACHE_DIR = "./embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# CLIP Model for embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Object segmentation model
seg_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True,
    token=huggingface_token
).to(device)

seg_processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True,
    token=huggingface_token
)

# JSON data
try:
    with open("perfume_image.json", "r", encoding="utf-8") as f:
        perfume_data = json.load(f)
        print(f"Loaded {len(perfume_data)} items from JSON.")
except Exception as e:
    print(f"Failed to load JSON file: {e}")
    perfume_data = []

# URL hash function
def generate_hash(url):
    return hashlib.md5(url.encode('utf-8')).hexdigest()

# Image download and caching
def download_image_with_cache(url):
    filename = os.path.basename(url.split("?")[0])
    local_path = os.path.join(CACHE_DIR, filename)

    if os.path.exists(local_path):
        print(f"Using cached image: {local_path}")
        return Image.open(local_path).convert("RGB")

    print(f"Downloading image: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(response.content)

    return Image.open(local_path).convert("RGB")

# Compute embeddings for an image
def compute_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    outputs = clip_model.get_image_features(**inputs)
    return outputs.cpu().detach().numpy()

# Crop and detect bottles using the segmentation model
def detect_and_crop_bottle(image):
    prompt = "<OD>"
    inputs = seg_processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = seg_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        num_beams=3,
        do_sample=False
    )
    generated_text = seg_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = seg_processor.post_process_generation(
        generated_text, task="<OD>", image_size=(image.width, image.height)
    )

    for bbox, label in zip(parsed_answer['<OD>']['bboxes'], parsed_answer['<OD>']['labels']):
        if label.lower() == "bottle":  # Only process bottles
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = image.crop((x1, y1, x2, y2))
            # cropped_image.show()
            return cropped_image
    return image

# Search API
@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Detect and crop the bottle
        cropped_image = detect_and_crop_bottle(image)
        # if cropped_image is None:
        #     return {"error": "No bottle detected in the uploaded image."}

        # Compute embedding for the cropped image
        embedding = compute_embedding(cropped_image).flatten()
        embedding /= np.linalg.norm(embedding)

        # Search using FAISS
        D, I = index.search(embedding.reshape(1, -1).astype(np.float32), k=10)

        # Filter results
        threshold = 0.3
        results = [
            {
                "index": int(i),
                "id": db_images[i]["id"],
                "url": db_images[i]["url"],
                "similarity": float(D[0][idx])
            }
            for idx, i in enumerate(I[0]) if float(D[0][idx]) > threshold
        ]

        return {"results": results}
    except Exception as e:
        print(f"Error during search: {e}")
        return {"error": str(e)}

# Initialize embeddings and FAISS index
db_images = []
db_embeddings = []

for item in perfume_data:
    try:
        image = download_image_with_cache(item["url"])
        db_images.append({"id": item["id"], "url": item["url"]})
        embedding = compute_embedding(image).flatten()
        db_embeddings.append(embedding)
    except Exception as e:
        print(f"Failed to process image ID {item['id']} from URL {item['url']}: {e}")

if db_embeddings:
    db_embeddings = np.array(db_embeddings)
    normalized_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings.astype(np.float32))
else:
    index = faiss.IndexFlatIP(1)  # Empty FAISS index
