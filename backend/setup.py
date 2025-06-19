from setuptools import setup, find_packages

setup(
    name="brand-detection-api",
    version="2.0.0",
    description="Enhanced Brand Detection API with OCR and NLP",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "pillow>=10.1.0",
        "numpy>=1.24.3",
        "google-cloud-vision>=3.4.5",
        "pytesseract>=0.3.10",
        "pydantic>=2.5.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)