import easyocr , cv2

reader = easyocr.Reader(['th','en'],gpu=False)
image_file = 'output_004.png'
result = reader.readtext(image_file)


min_confidence = 0.5
filtered_results = [res for res in result if res[2] >= min_confidence]

for _, text, conf in filtered_results:
    print(f"{text} (Confidence: {conf:.2f})")
    
image = cv2.imread(image_file)
for(bbox,text,conf) in filtered_results:
    (top_left,top_right,bottom_right,bottom_left) = bbox
    top_left=tuple(map(int,top_left))
    bottom_right = tuple(map(int,bottom_right))
    cv2.rectangle(image,top_left,bottom_right,(0,255,0),2)
cv2.imwrite("output_004.jpg",image)