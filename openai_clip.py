from fastapi import FastAPI, File, UploadFile
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io, os, requests, json, hashlib, torch, faiss
from rembg import remove

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = FastAPI()

# 캐시 디렉토리 설정
CACHE_DIR = "./image_cache"
EMBEDDING_CACHE_DIR = "./embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# CLIP 모델 및 프로세서 로드 (GPU로 설정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# JSON 데이터 로드
try:
    with open("perfume_image.json", "r", encoding="utf-8") as f:
        perfume_data = json.load(f)
        print(f"Loaded {len(perfume_data)} items from JSON.")
except Exception as e:
    print(f"Failed to load JSON file: {e}")
    perfume_data = []

# URL 해시 생성 함수
def generate_hash(url):
    return hashlib.md5(url.encode('utf-8')).hexdigest()

# URL에서 이미지를 다운로드하고 캐싱하는 함수
def download_image_with_cache(url):
    # URL에서 파일 이름과 확장자 추출
    filename = os.path.basename(url.split("?")[0])  # 쿼리 파라미터 제거
    local_path = os.path.join(CACHE_DIR, filename)

    # 캐시에 이미지가 이미 있는지 확인
    if os.path.exists(local_path):
        print(f"Using cached image: {local_path}")
        return Image.open(local_path).convert("RGB")

    # 이미지 다운로드
    print(f"Downloading image: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # 이미지 저장
    with open(local_path, "wb") as f:
        f.write(response.content)

    # 로컬에서 이미지 열기
    return Image.open(local_path).convert("RGB")

# 임베딩 생성 함수
def compute_embedding(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.get_image_features(**inputs)
    return outputs.cpu().detach().numpy()  # GPU에서 CPU로 변환

# 임베딩 캐싱 함수
def get_or_compute_embedding(image, url):
    # URL 기반으로 해시 생성
    url_hash = generate_hash(url)
    embedding_path = os.path.join(EMBEDDING_CACHE_DIR, f"{url_hash}.npy")

    # 캐시된 임베딩 파일이 존재하면 불러오기
    if os.path.exists(embedding_path):
        print(f"Using cached embedding for URL: {url}")
        return np.load(embedding_path)

    # 임베딩 생성
    print(f"Computing embedding for URL: {url}")
    embedding = compute_embedding(image).flatten()

    # 임베딩 저장
    np.save(embedding_path, embedding)
    return embedding

# 데이터베이스 이미지 임베딩 생성
db_images = []
db_embeddings = []

for item in perfume_data:
    try:
        # 이미지 다운로드 및 캐싱
        image = download_image_with_cache(item["url"])
        db_images.append({"id": item["id"], "url": item["url"]})

        # 임베딩 생성 또는 캐싱된 값 불러오기
        embedding = get_or_compute_embedding(image, item["url"])
        db_embeddings.append(embedding)
    except Exception as e:
        print(f"Failed to process image ID {item['id']} from URL {item['url']}: {e}")

# NumPy 배열로 변환 및 FAISS 인덱스 초기화
if db_embeddings:
    db_embeddings = np.array(db_embeddings)
    print(f"Embeddings array shape: {db_embeddings.shape}")

    # 코사인 유사도를 위한 FAISS 인덱스 생성
    dimension = db_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine Similarity 기반
    normalized_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    index.add(normalized_embeddings.astype(np.float32))  # float32로 변환하여 FAISS에 추가
else:
    print("No embeddings were generated. Initializing empty FAISS index.")
    index = faiss.IndexFlatIP(1)  # 빈 인덱스 생성

# 업로드된 이미지의 임베딩 계산 함수
def compute_uploaded_image_embedding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embedding = compute_embedding(image).flatten()
    return embedding

# 검색 API 엔드포인트
@app.post("/get_similarity/")
async def search_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        output_image_bytes = remove(image_bytes.getvalue())

        output_image = Image.open(io.BytesIO(output_image_bytes)).convert("RGBA")

        # 배경 흰색으로 설정
        white_background = Image.new("RGBA", output_image.size, "WHITE")
        white_background.paste(output_image, mask=output_image)

        final_image = white_background.convert("RGB")
        # final_image.show()

        # 업로드된 이미지 임베딩 계산
        query_embedding = compute_embedding(final_image).flatten()
        query_embedding /= np.linalg.norm(query_embedding)  # 정규화

        # FAISS 인덱스에서 검색
        D, I = index.search(query_embedding.reshape(1, -1).astype(np.float32), k=10)

        # 결과 생성
        threshold = 0.3  # 유사도 임계값
        results = [
            {
                "index": int(i),
                "id": db_images[i]["id"],
                "url": db_images[i]["url"],
                "similarity": float(D[0][idx])  # 코사인 유사도 점수
            }
            for idx, i in enumerate(I[0]) if float(D[0][idx]) > threshold
        ]

        return {"results": results}
    except Exception as e:
        print(f"Error during search: {e}")
        return {"error": str(e)}