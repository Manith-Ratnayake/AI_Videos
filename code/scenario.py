import cv2

def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % (fps * interval) == 0:
            frames.append(frame)
        
        frame_count += 1

    cap.release()
    return frames




from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Pre-trained model

def detect_objects(frame):
    print("\nModel- ",model)
    results = model(frame)
    # objects = [result.names[int(cls)] for result in results for cls in result.boxes.cls]

    print("results ---", results)

    for result in results:
        classes = results.boxes.cls.cpu().numpy()

        for cls_id in classes:
            objects.append(model.name[int(cls_id)])


    print("---- " , objects)
    return objects


from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    image = Image.fromarray(image)
    inputs = processor(images=image, return_tensors="pt")
    caption = model.generate(**inputs)
    return processor.decode(caption[0], skip_special_tokens=True)



from transformers import pipeline

summary_model = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_captions(captions):
    text = " ".join(captions)
    summary = summary_model(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]


video_path = "data/a.mp4"
frames = extract_frames(video_path, interval=2)

captions = []
for frame in frames:
    print("frame -- ", frame)
    objects = detect_objects(frame)
    caption = generate_caption(frame)
    captions.append(f"{caption}. Detected objects: {', '.join(objects)}.")

video_summary = summarize_captions(captions)
print("Video Summary:", video_summary)
