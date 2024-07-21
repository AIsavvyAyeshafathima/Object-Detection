from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform object detection
def detect_objects(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detections.append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "box": box
        })
    return detections

# Function to identify dominant colors
def get_dominant_colors(image_url, k=5):
    response = requests.get(image_url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        print("Error: Could not load image. Please check the image path.")
        return None, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    label_counts = np.bincount(labels.flatten())
    total_pixels = len(labels.flatten())
    percentages = label_counts / total_pixels * 100

    return centers, percentages

# Combined function to perform both tasks
def analyze_image(image_url, k=5):
    # Detect objects
    detections = detect_objects(image_url)
    print("Detecting objects:-")
    if not detections:
        print("No objects detected.")
    else:
        for detection in detections:
            print(f"Detected {detection['label']} with confidence {detection['confidence']} at location {detection['box']}")

    print("\nIdentifying dominant colors:-")
    dominant_colors, percentages = get_dominant_colors(image_url, k)
    if dominant_colors is not None and percentages is not None:
        print("Dominant colors (RGB):")
        print(dominant_colors)
        print("Percentages:", percentages)

        for i, (color, percentage) in enumerate(zip(dominant_colors, percentages)):
            plt.subplot(1, k, i+1)
            plt.imshow(np.ones((100, 100, 3), dtype=np.uint8) * color)
            plt.title(f'{percentage:.2f}%')
            plt.axis('off')
        plt.show()

# Example usage
image_url = "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/4082x3456+0+0/resize/1600/quality/85/format/webp/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2F5a%2F9f%2F1f84805d410e8c1c8416c6caac8a%2Fvocal-crow-4.jpg"
analyze_image(image_url, k=5)