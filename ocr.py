import easyocr

reader = easyocr.Reader(['en'])

data = 'silver.jpg'

result = reader.readtext(data, detail=0)
print(result)