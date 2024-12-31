import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# CLIP 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 이미지에서 특징 벡터 추출
def compute_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy()

# 데이터베이스 이미지 특징 벡터 사전 계산
db_images = ["바이레도모하비고스트오드퍼퓸.jpg", 
             "바이레도블랑쉬오드퍼퓸.jpg", 
             "이솝테싯오드퍼퓸.jpg", 
             "이솝글롬.png",
             "이솝로즈.jpg",
             "이솝미라세티.png",
             "이솝우라논.png",
             "이솝이더시스오드퍼퓸.jpg",
             "이솝카르스트.png",
             "이솝마라케시.png",
             "이솝휠오드퍼퓸.jpg"]
db_embeddings = np.array([compute_embedding(img).flatten() for img in db_images])

# FAISS 인덱스 생성 및 벡터 추가
dimension = db_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 거리 기반
index.add(db_embeddings)

# 업로드된 이미지 검색
query_image = "aesop_tacit.jpeg"
query_embedding = compute_embedding(query_image).flatten()
D, I = index.search(query_embedding.reshape(1, -1), k=5)  # 상위 5개 검색

# 결과 출력
print("유사도 거리:", D)
print("매칭된 이미지 인덱스:", I)
