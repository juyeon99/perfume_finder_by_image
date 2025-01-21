from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModel
from PIL import Image
import io, os, hashlib, torch, requests, logging
from rembg import remove
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device)

# 캐시 디렉토리 설정
CACHE_DIR = "../image_cache"
EMBEDDING_CACHE_DIR = "../jina_embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

app = FastAPI()

# URL 해시 생성 함수
def generate_hash(url):
    return hashlib.md5(url.encode("utf-8")).hexdigest()

# URL에서 이미지를 다운로드하고 캐싱하는 함수
def download_image_with_cache(url):
    filename = os.path.basename(url.split("?")[0])  # 쿼리 파라미터 제거
    local_path = os.path.join(CACHE_DIR, filename)

    if os.path.exists(local_path):
        logger.info(f"Using cached image: {local_path}")
        return Image.open(local_path).convert("RGB")

    logger.info(f"Downloading image: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(response.content)

    return Image.open(local_path).convert("RGB")

def get_or_compute_embedding(image, url):
    # URL 기반으로 해시 생성
    url_hash = generate_hash(url)
    embedding_path = os.path.join(EMBEDDING_CACHE_DIR, f"{url_hash}.npy")

    # 캐시된 임베딩 파일이 존재하면 불러오기
    if os.path.exists(embedding_path):
        logger.info(f"Using cached embedding for URL: {url}")
        return torch.tensor(np.load(embedding_path))

    # 임베딩 생성
    logger.info(f"Computing embedding for URL: {url}")
    embedding = model.encode_image(image).flatten()

    # 임베딩 저장
    np.save(embedding_path, embedding)
    return embedding

# GPU에서 실행되는 임베딩 생성 함수
@app.post("/compute_embedding/")
async def compute_embedding(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 배경 제거
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        output_image_bytes = remove(image_bytes.getvalue())

        output_image = Image.open(io.BytesIO(output_image_bytes)).convert("RGBA")

        white_background = Image.new("RGBA", output_image.size, "WHITE")
        white_background.paste(output_image, mask=output_image)
        final_image = white_background.convert("RGB")

        # 임베딩 계산
        embedding = model.encode_image(final_image).flatten()
        embedding = torch.tensor(embedding)
        embedding = embedding / embedding.norm()

        return {"embedding": embedding.cpu().detach().numpy().tolist()}
    except Exception as e:
        return {"error": str(e)}
