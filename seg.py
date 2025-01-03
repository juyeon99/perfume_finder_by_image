import requests
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM 
from huggingface_hub import login
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

login(huggingface_token)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 모델 로드 시 인증 토큰 사용
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large", 
    torch_dtype=torch_dtype, 
    trust_remote_code=True,
    use_auth_token=huggingface_token
).to(device)

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large", 
    trust_remote_code=True,
    use_auth_token=huggingface_token
)

prompt = "<OD>"
url = "https://cf.bysuco.net/86a60a5843ff79675732d6a0091cd04a.jpg?w=600&f=jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=4096,
    num_beams=3,
    do_sample=False
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
print(parsed_answer)

# 시각화 부분
draw_image = image.copy()
draw = ImageDraw.Draw(draw_image)

# parsed_answer에서 직접 정보 사용
for bbox, label in zip(parsed_answer['<OD>']['bboxes'], parsed_answer['<OD>']['labels']):
    x1, y1, x2, y2 = bbox
    
    # 빨간색 박스 그리기 (두께 2)
    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
    
    # 라벨 텍스트 그리기
    draw.text((x1, y1-20), label, fill='red')

# 결과 이미지 저장
draw_image.save('detected_objects.jpg')

# 결과 이미지 보여주기
draw_image.show()