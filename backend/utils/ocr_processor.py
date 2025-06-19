from google.cloud import vision
import os

def extract_text_from_image(image_path: str) -> str:
    """Extracts text from an image using Google Vision API."""
    client = vision.ImageAnnotatorClient()
    
    with open(image_path, "rb") as f:
        image = vision.Image(content=f.read())
    
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description
    return ""