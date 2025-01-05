from fastapi import FastAPI, File, UploadFile
import faiss
import numpy as np
from transformers import AutoModel
from PIL import Image, ImageEnhance
import io,os,json,hashlib,torch,re
from rembg import remove

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

CACHE_DIR = "./image_cache"
EMBEDDING_CACHE_DIR = "./jina_embedding_cache"
TEST_IMAGE_DIR = "./test"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)

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
    filename = os.path.basename(url.split("?")[0])
    local_path = os.path.join(CACHE_DIR, filename)

    if os.path.exists(local_path):
        return Image.open(local_path).convert("RGB")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(response.content)

    return Image.open(local_path).convert("RGB")

# 밝기를 조정하는 함수
def enhance_image_brightness(image, factor=2.0):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

# 임베딩 생성 함수
def compute_embedding(image):
    return model.encode_image(image)

# 임베딩 캐싱 함수
def get_or_compute_embedding(image, url):
    url_hash = generate_hash(url)
    embedding_path = os.path.join(EMBEDDING_CACHE_DIR, f"{url_hash}.npy")

    if os.path.exists(embedding_path):
        return np.load(embedding_path)

    embedding = compute_embedding(image).flatten()
    np.save(embedding_path, embedding)
    return embedding

# 데이터베이스 이미지 임베딩 생성
db_images = []
db_embeddings = []

for item in perfume_data:
    try:
        image = download_image_with_cache(item["url"])
        db_images.append({"id": item["id"], "url": item["url"]})
        print(f"Processing image ID {item['id']}")

        embedding = get_or_compute_embedding(image, item["url"])
        db_embeddings.append(embedding)
    except Exception as e:
        print(f"Failed to process image ID {item['id']} from URL {item['url']}: {e}")

if db_embeddings:
    db_embeddings = np.array(db_embeddings)
    dimension = db_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    normalized_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    index.add(normalized_embeddings.astype(np.float32))
else:
    index = faiss.IndexFlatIP(1)

def evaluate():
    test_results = []

    for test_image_name in os.listdir(TEST_IMAGE_DIR):
        test_image_path = os.path.join(TEST_IMAGE_DIR, test_image_name)
        try:
            image = Image.open(test_image_path).convert("RGB")
            # image = enhance_image_brightness(image, factor=2.0)

            cropped_image = image
            cropped_image_bytes = io.BytesIO()
            cropped_image.save(cropped_image_bytes, format="PNG")
            cropped_image_bytes.seek(0)
            output_image_bytes = remove(cropped_image_bytes.getvalue())

            # Convert the output bytes to a PIL image
            output_image = Image.open(io.BytesIO(output_image_bytes)).convert("RGBA")

            # Add a white background
            white_background = Image.new("RGBA", output_image.size, "WHITE")
            white_background.paste(output_image, mask=output_image)

            # Convert the image to RGB format for embedding computation
            final_image = white_background.convert("RGB")

            embedding = compute_embedding(final_image).flatten()
            embedding /= np.linalg.norm(embedding)

            D, I = index.search(embedding.reshape(1, -1).astype(np.float32), k=1)

            results = [
                {
                    "index": int(i),
                    "id": db_images[i]["id"],
                    "url": db_images[i]["url"],
                    "similarity": float(D[0][idx])
                }
                for idx, i in enumerate(I[0])
            ]

            # 정답과 비교 (정확도 평가)
            match = re.search(r"\d+", test_image_name)  # 파일명에서 숫자 추출
            ground_truth_id = match.group(0) if match else None  # 정답 ID 설정
            if ground_truth_id:
                correct = any(str(result["id"]) == ground_truth_id for result in results)
            else:
                correct = False

            test_results.append({
                "test_image": test_image_name,
                "ground_truth_id": ground_truth_id,
                "correct": correct,
                "results": results
            })
            print(f"Processed test result: {test_results}")
        except Exception as e:
            print(f"Failed to process test image {test_image_name}: {e}")

    accuracy = sum(1 for result in test_results if result["correct"]) / len(test_results)
    true_count = sum(1 for result in test_results if result["correct"])
    false_count = len(test_results) - true_count

    return {
        "accuracy": accuracy,
        "true_count": true_count,
        "false_count": false_count,
        "details": test_results
    }

if __name__ == "__main__":
    evaluation_results = evaluate()
    accuracy = evaluation_results["accuracy"]  # 정확도 값을 가져옵니다.
    true_count = evaluation_results["true_count"]
    false_count = evaluation_results["false_count"]

    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
    print(f"True Matches: {true_count}, False Matches: {false_count}")
    for result in evaluation_results["details"]:
        print(f"Test Image: {result['test_image']}, Ground Truth ID: {result['ground_truth_id']}, Correct: {result['correct']}")