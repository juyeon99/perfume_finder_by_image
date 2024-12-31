# from fastapi import FastAPI, File, UploadFile
# import faiss
# import numpy as np
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import io
# import os

# # Set environment variable for KMP_DUPLICATE_LIB_OK
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# # Initialize FastAPI app
# app = FastAPI()

# # Load the CLIP model and processor
# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# # Database images
# db_images = [
#     "바이레도모하비고스트오드퍼퓸.jpg", 
#     "바이레도블랑쉬오드퍼퓸.jpg", 
#     "이솝테싯오드퍼퓸.jpg", 
#     "이솝글롬.png",
#     "이솝로즈.jpg",
#     "이솝미라세티.png",
#     "이솝우라논.png",
#     "이솝이더시스오드퍼퓸.jpg",
#     "이솝카르스트.png",
#     "이솝마라케시.png",
#     "이솝휠오드퍼퓸.jpg"
# ]

# # Compute image embeddings function
# def compute_embedding(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt")
#     outputs = model.get_image_features(**inputs)
#     return outputs.detach().numpy()

# # Precompute the embeddings for the database images
# db_embeddings = np.array([compute_embedding(img).flatten() for img in db_images])

# # Create the FAISS index
# dimension = db_embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(db_embeddings)

# # Helper function to compute embedding from uploaded image
# def compute_uploaded_image_embedding(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt")
#     outputs = model.get_image_features(**inputs)
#     return outputs.detach().numpy()

# @app.post("/search/")
# async def search_image(file: UploadFile = File(...)):
#     image_bytes = await file.read()
    
#     # Compute the embedding of the uploaded image
#     query_embedding = compute_uploaded_image_embedding(image_bytes).flatten()

#     # Search for the most similar images using FAISS
#     D, I = index.search(query_embedding.reshape(1, -1), k=5)

#     # Prepare results with proper type casting (int, float)
#     results = [
#         {"index": int(i), "image": db_images[i], "distance": float(D[0][idx])}  # Ensure int and float
#         for idx, i in enumerate(I[0])
#     ]
    
#     return {"results": results}


from fastapi import FastAPI, File, UploadFile
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import os
import requests
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = FastAPI()

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load perfume image data from JSON
with open("perfume_image.json", "r", encoding="utf-8") as f:
    perfume_data = json.load(f)

# Helper function to download image from a URL
def download_image(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return image

# Compute image embeddings function
def compute_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy()

# Precompute the embeddings for the database images
db_images = []
db_embeddings = []

for item in perfume_data:
    try:
        # Download the image
        image = download_image(item["url"])
        db_images.append({"id": item["id"], "url": item["url"]})
        
        # Compute and store the embedding
        embedding = compute_embedding(image).flatten()
        db_embeddings.append(embedding)
    except Exception as e:
        print(f"Failed to process image ID {item['id']} from URL {item['url']}: {e}")

# Convert embeddings to a NumPy array
db_embeddings = np.array(db_embeddings)

# Create the FAISS index
dimension = db_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(db_embeddings)

# Helper function to compute embedding from uploaded image
def compute_uploaded_image_embedding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy()

@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Compute the embedding of the uploaded image
    query_embedding = compute_uploaded_image_embedding(image_bytes).flatten()

    # Search for the most similar images using FAISS
    D, I = index.search(query_embedding.reshape(1, -1), k=3)

    # Prepare results with proper type casting (int, float)
    results = [
        {
            "index": int(i),
            "id": db_images[i]["id"],
            "url": db_images[i]["url"],
            "distance": float(D[0][idx])
        }
        for idx, i in enumerate(I[0])
    ]
    
    return {"results": results}
