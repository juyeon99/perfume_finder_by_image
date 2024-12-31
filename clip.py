from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# CLIP 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 향수 이름 및 설명 (예시)
perfumes = [
    "Dior Sauvage perfume bottle",
    "Chanel No. 5 perfume bottle",
    "Tom Ford Black Orchid perfume bottle",
    "Gucci Guilty Pour Femme eau de parfum",
    "Gucci Guilty Pour Femme eau de toilette",
    "Aesop Marrakech Intense Eau de Parfum",
    "Aesop Hwyl Eau de Parfum",
    "diptyque fleur de peau edp",
    "diptyque orpheon Eau de Parfum",
    "Creed SILVER MOUNTAIN WATER Eau de Parfum",
    "Creed Aventus Eau de Parfum",
]

# 텍스트 벡터 생성
text_inputs = processor(text=perfumes, return_tensors="pt", padding=True)
text_features = model.get_text_features(**text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)  # 정규화

# 업로드된 이미지 처리
image = Image.open("fleur2.jpg").convert("RGB")
image_inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**image_inputs)
image_features /= image_features.norm(dim=-1, keepdim=True)  # 정규화

# 유사도 계산
similarities = torch.matmul(image_features, text_features.T)
best_match_idx = similarities.argmax().item()

print(f"Similarities: {similarities.flatten()}")
print(f"Best match: {perfumes[best_match_idx]}")
