from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import model_service
import requests, faiss, json, torch, io, os, logging
import numpy as np
from contextlib import asynccontextmanager

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 로그
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DB, FAISS 인덱스 초기화
db_images = []
db_embeddings = []
index = None
perfume_data = []

# 서버 시작 전 미리 실행할 코드
def init():
    global db_images, db_embeddings, index, perfume_data  # Declare as global to modify the global variable
    
    # JSON 향수 정보 로드
    try:
        # 향수 이미지 로드
        with open("../perfume_image.json", "r", encoding="utf-8") as f:
            perfume_image_data = json.load(f)
            logger.info(f"Loaded {len(perfume_image_data)} perfume_image_datas from JSON.")
        # 향수 정보 로드
        with open("../perfume.json", "r", encoding="utf-8") as f:
            perfume_data = json.load(f)
            logger.info(f"Loaded {len(perfume_data)} perfume_datas from JSON.")
    except Exception as e:
        logger.error(f"Failed to load JSON files: {str(e)}")
        perfume_image_data = []
        perfume_data = []

    # 이미지 임베딩 생성
    for item in perfume_image_data:
        try:
            image = model_service.download_image_with_cache(item["url"])
            db_images.append({"id": item["id"], "url": item["url"]})
            embedding = model_service.get_or_compute_embedding(image, item["url"])
            db_embeddings.append(embedding)
        except Exception as e:
            logger.error(f"Failed to process image ID {item['id']} from URL {item['url']}: {e}")

    # FAISS 인덱스 생성
    if db_embeddings:
        db_embeddings = torch.stack(db_embeddings)
        db_embeddings = db_embeddings / db_embeddings.norm(dim=1, keepdim=True)

        dimension = db_embeddings.shape[1]
        index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), dimension)  # Use global index here
        index.add(db_embeddings.numpy())
    else:
        logger.info("No embeddings generated. Initializing empty FAISS index.")
        index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), 1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 임베딩값으로 향수 매칭
def get_matching_perfumes(embedding, db_images, db_embeddings, perfume_data, threshold=0.3):
    # FAISS 인덱스에서 검색
    D, I = index.search(np.array(embedding).reshape(1, -1), k=10)

    results = [
        {
            "index": int(i),
            "id": db_images[i]["id"],
            "url": db_images[i]["url"],
            "similarity": float(D[0][idx]),
        }
        for idx, i in enumerate(I[0])
        if float(D[0][idx]) > threshold
    ]

    ids = [result["id"] for result in results]
    matching_perfumes = [
        {
            "id": item["id"],
            "name": item["name"],
            "brand": item["brand"],
            "description": item["description"],
            "similarity": next(
                (result["similarity"] for result in results if result["id"] == item["id"]), None
            ),
            "url": next(
                (result["url"] for result in results if result["id"] == item["id"]), None
            ),
        }
        for item in perfume_data if item["id"] in ids
    ]

    return sorted(matching_perfumes, key=lambda x: x["similarity"], reverse=True)

@app.post("/get_perfume_details/")
async def search_image(file: UploadFile = File(...)):
    try:
        # GPU 임베딩 서비스 호출
        image_bytes = await file.read()

        compute_url = "http://localhost:8001/compute_embedding/"
        response = requests.post(
            compute_url, files={"file": ("uploaded_image.png", image_bytes)}
        )

        if response.status_code == 200:
            embedding = response.json().get("embedding")
            
            if embedding is not None:
                matching_perfumes = get_matching_perfumes(embedding, db_images, db_embeddings, perfume_data)

                return {"perfumes": sorted(matching_perfumes, key=lambda x: x["similarity"], reverse=True)}
            else:
                return {"error": "No embedding found in the response"}
        else:
            return {"error": f"Failed to get embedding. Status code: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error retrieving perfume details: {e}")
        return {"error": str(e)}
