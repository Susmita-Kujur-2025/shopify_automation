import pytest
from fastapi.testclient import TestClient
from api.main import app
import json
import io
from PIL import Image

client = TestClient(app)

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (200, 100), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_detect_endpoint():
    """Test detection endpoint"""
    # Create test data
    test_image = create_test_image()
    
    brands = [
        {
            "name": "TestBrand",
            "description": "Test brand",
            "product_types": ["Tea", "Coffee"],
            "aliases": ["TB"],
            "keywords": ["organic"]
        }
    ]
    
    categories = ["Herbal Tea", "Coffee"]
    
    # Make request
    response = client.post(
        "/api/detect",
        files={"image": ("test.jpg", test_image, "image/jpeg")},
        data={
            "brands": json.dumps(brands),
            "product_categories": json.dumps(categories)
        }
    )
    
    # Check response
    assert response.status_code in [200, 400]  # 400 if no text found