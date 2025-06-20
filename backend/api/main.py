# Enhanced Brand Detection Backend for Vercel
# File: api/main.py (Vercel requires this structure)

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import base64
import io
import json
import re
from PIL import Image
import numpy as np
import os
from openai import OpenAI


# For Vercel deployment, we'll use lighter alternatives
try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    # Fallback to pytesseract for OCR
    import pytesseract

# Enhanced fuzzy matching
from difflib import SequenceMatcher
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Enhanced Brand Detection API", version="2.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
openAIkey = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openAIkey) if openAIkey else None

# Enhanced models
class BrandInfo(BaseModel):
    name: str
    description: Optional[str] = None
    aliases: Optional[List[str]] = []
    product_types: Optional[List[str]] = []

class DetectionResponse(BaseModel):
    brand: str
    best_product: str
    confidence: float
    extracted_text: str
    processing_metadata: Dict[str, Any]
    renamed_file: Optional[Dict[str, str]] = None

class EnhancedOCRProcessor:
    """Enhanced OCR with multiple fallbacks and preprocessing"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def preprocess_image(self, image_bytes: bytes) -> bytes:
        """Enhance image quality for better OCR"""
        try:
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast and brightness
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            # Resize if too large (for faster processing)
            max_size = 1500
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert back to bytes
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=95)
            return output.getvalue()
            
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return image_bytes
    
    async def extract_text_google_vision(self, image_bytes: bytes) -> str:
        """Extract text using Google Vision API"""
        if not GOOGLE_VISION_AVAILABLE:
            return ""
        
        try:
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=image_bytes)
            
            # Use both text detection and document text detection
            response = client.text_detection(image=image)
            doc_response = client.document_text_detection(image=image)
            
            texts = []
            if response.text_annotations:
                texts.append(response.text_annotations[0].description)
            
            if doc_response.full_text_annotation:
                texts.append(doc_response.full_text_annotation.text)
            
            # Combine and deduplicate
            combined_text = " ".join(texts)
            return self.clean_text(combined_text)
            
        except Exception as e:
            print(f"Google Vision failed: {e}")
            return ""
    
    def extract_text_tesseract(self, image_bytes: bytes) -> str:
        """Fallback OCR using Tesseract"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Try different OCR configurations
            configs = [
                '--oem 3 --psm 6',  # Default
                '--oem 3 --psm 8',  # Single word
                '--oem 3 --psm 7',  # Single text line
                '--oem 3 --psm 11', # Sparse text
            ]
            
            results = []
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    if text.strip():
                        results.append(text)
                except:
                    continue
            
            # Return the longest result
            if results:
                best_result = max(results, key=len)
                return self.clean_text(best_result)
            
            return ""
            
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    async def extract_text(self, image_bytes: bytes) -> str:
        """Main OCR function with multiple methods"""
        # Preprocess image
        processed_image = self.preprocess_image(image_bytes)
        
        # Try Google Vision first
        text = await self.extract_text_google_vision(processed_image)
        
        # Fallback to Tesseract if Google Vision fails
        if not text and 'pytesseract' in globals():
            text = self.extract_text_tesseract(processed_image)
        
        return text or ""

class EnhancedNLPParser:
    """Enhanced NLP with fuzzy matching and better confidence scoring"""
    
    def __init__(self):
        self.common_product_words = {
            'tea', 'coffee', 'capsule', 'tablet', 'powder', 'oil', 'cream',
            'lotion', 'serum', 'supplement', 'vitamin', 'protein', 'juice',
            'syrup', 'extract', 'tincture', 'drops', 'gel', 'balm', 'soap',
            'mushroom', 'mushrooms', 'herbal', 'natural', 'organic'
        }
    
    def fuzzy_match(self, text: str, candidates: List[str], threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Fuzzy string matching with confidence scores"""
        matches = []
        text_lower = text.lower().strip()
        
        for candidate in candidates:
            candidate_lower = candidate.lower().strip()
            
            # Exact match
            if candidate_lower == text_lower:
                matches.append({
                    'match': candidate,
                    'confidence': 1.0,
                    'method': 'exact'
                })
                continue
            
            # Normalize spaces and special characters
            text_normalized = ' '.join(text_lower.split())
            candidate_normalized = ' '.join(candidate_lower.split())
            
            # Check if candidate is a subset of text or vice versa
            if candidate_normalized in text_normalized or text_normalized in candidate_normalized:
                ratio = len(candidate_normalized) / max(len(text_normalized), len(candidate_normalized))
                if ratio >= threshold:
                    matches.append({
                        'match': candidate,
                        'confidence': ratio,
                        'method': 'subset'
                    })
                    continue
            
            # Fuzzy match using sequence matcher
            ratio = SequenceMatcher(None, candidate_normalized, text_normalized).ratio()
            if ratio >= threshold:
                matches.append({
                    'match': candidate,
                    'confidence': ratio,
                    'method': 'fuzzy'
                })
                continue
            
            # Word-level matching
            candidate_words = set(candidate_normalized.split())
            text_words = set(text_normalized.split())
            
            if candidate_words.issubset(text_words):
                overlap_ratio = len(candidate_words.intersection(text_words)) / len(candidate_words)
                if overlap_ratio >= threshold:
                    matches.append({
                        'match': candidate,
                        'confidence': overlap_ratio * 0.9,  # Slightly lower than exact
                        'method': 'word_match'
                    })
        
        return sorted(matches, key=lambda x: x['confidence'], reverse=True)
    
    async def extract_product_with_openai(self, text: str, brand_info: BrandInfo) -> Dict[str, Any]:
        """Use OpenAI to extract product information"""
        try:
            # Prepare the prompt
            prompt = f"""Given the following text extracted from a product image and brand information, identify the specific product name.

Brand Information:
- Name: {brand_info.name}
- Description: {brand_info.description}
- Product Types: {', '.join(brand_info.product_types) if brand_info.product_types else 'Not specified'}

Extracted Text:
{text}

Please identify the specific product name from the text. Consider the brand's description and product types.
Return the response in JSON format with the following structure:
{{
    "product_name": "The specific product name",
    "confidence": A number between 0 and 1 indicating confidence,
    "explanation": "Brief explanation of why this is the product name"
}}

Only return the JSON, no additional text."""

            response = client.responses.create(
                model="gpt-4o",
                input = (
                    f"You are a product detection assistant that identifies specific product names from text. "
                    f"The brand name is {brand_info.name}, brand description is {brand_info.description}, "
                    f"the extracted text is {text}."
                )
            )


            # Parse the response
            result = response.output_text
            print(f"response from openAI: {response}")
            return {
                'best_product': result['product_name'],
                'product_confidence': result['confidence'],
                'explanation': result['explanation']
            }

        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback to basic product detection
            return self.extract_product_info(text, brand_info.name)
    
    def extract_brand_names(self, text: str, brand_info: BrandInfo) -> List[Dict[str, Any]]:
        """Extract brand names with enhanced matching"""
        print(f"\nBrand Detection Log:")
        print(f"Looking for brand: {brand_info.name}")
        print(f"With variations: {brand_info.aliases}")
        print(f"In text: {text}")
        
        # First try exact match with main brand name
        main_brand_match = self.fuzzy_match(text, [brand_info.name], threshold=0.7)
        if main_brand_match:
            print(f"Found main brand match: {main_brand_match[0]}")
            return [{
                'brand': brand_info.name,
                'confidence': main_brand_match[0]['confidence'],
                'matched_text': main_brand_match[0]['match'],
                'method': main_brand_match[0]['method']
            }]
        
        # If no main brand match, try with variations
        if brand_info.aliases:
            # Try each variation individually
            for variation in brand_info.aliases:
                variation_match = self.fuzzy_match(text, [variation], threshold=0.7)
                if variation_match:
                    print(f"Found variation match: {variation_match[0]}")
                    return [{
                        'brand': brand_info.name,  # Still return main brand name
                        'confidence': variation_match[0]['confidence'],
                        'matched_text': variation_match[0]['match'],
                        'method': 'variation'
                    }]
        
        # Last resort: Simple text containment check
        text_lower = text.lower()
        # Check main brand name
        if brand_info.name.lower() in text_lower:
            print(f"Found main brand through simple containment")
            return [{
                'brand': brand_info.name,
                'confidence': 0.5,  # Lower confidence for simple containment
                'matched_text': brand_info.name,
                'method': 'simple_containment'
            }]
        
        # Check variations
        if brand_info.aliases:
            for variation in brand_info.aliases:
                if variation.lower() in text_lower:
                    print(f"Found variation through simple containment: {variation}")
                    return [{
                        'brand': brand_info.name,
                        'confidence': 0.4,  # Even lower confidence for variation containment
                        'matched_text': variation,
                        'method': 'variation_containment'
                    }]
        
        print("No brand match found")
        return []
    
    def extract_product_info(self, text: str, brand_match: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced product extraction with context-aware parsing"""
        # Step 1: Find candidate phrases using context-aware patterns
        candidate_phrases = self.find_candidate_phrases(text, brand_match)
        
        # Step 2: Score and select best candidate
        best_candidate = self.select_best_candidate(candidate_phrases, text)
        
        return {
            'best_product': best_candidate or None,
            'product_confidence': 0.8 if best_candidate else 0.0,
            'all_products': candidate_phrases
        }
    
    def find_candidate_phrases(self, text: str, brand_match: Optional[str]) -> List[str]:
        """Find potential product names using multiple strategies"""
        candidates = set()
        
        # Strategy 1: Text immediately after brand name
        if brand_match:
            after_brand = self.extract_after_brand(text, brand_match)
            if after_brand:
                candidates.add(after_brand)
        
        # Strategy 2: Title-case phrases (2-4 words)
        candidates.update(self.extract_title_case_phrases(text))
        
        # Strategy 3: Repeated phrases (common in product labels)
        candidates.update(self.find_repeated_phrases(text))
        
        return sorted(candidates, key=len, reverse=True)
    
    def find_repeated_phrases(self, text: str) -> List[str]:
        """Identify repeated phrases (common in product packaging)"""
        words = re.split(r'\s+', text)
        phrases = {}
        
        # Generate 2-4 word sequences
        for n in (2, 3, 4):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                phrases[phrase] = phrases.get(phrase, 0) + 1
        
        # Return phrases appearing at least twice
        return [p for p, count in phrases.items() if count > 1 and len(p) > 5]
    
    def select_best_candidate(self, candidates: List[str], full_text: str) -> Optional[str]:
        """Select best candidate using scoring heuristics"""
        if not candidates:
            return None
        
        scored = []
        for candidate in candidates:
            score = 0.0
            
            # Position score (earlier is better)
            pos = full_text.find(candidate)
            if pos >= 0:
                score += max(0, 1.0 - (pos / len(full_text)))
            
            # Length score (longer is better)
            score += min(0.3, len(candidate) * 0.05)
            
            # Capitalization score
            if candidate.istitle():
                score += 0.2
                
            # Distinctiveness score
            common_words = {'natural', 'organic', 'premium', 'quality', 'original', 'supplement', 'dietary'}
            if not any(word in candidate.lower() for word in common_words):
                score += 0.3
                
            scored.append((candidate, score))
        
        # Return highest scored candidate
        return max(scored, key=lambda x: x[1])[0]
    
    def extract_after_brand(self, text: str, brand: str) -> Optional[str]:
        """Extract first meaningful phrase after brand name"""
        try:
            # Find brand position (case-insensitive)
            brand_regex = re.compile(re.escape(brand), re.IGNORECASE)
            match = brand_regex.search(text)
            if not match:
                return None
                
            # Extract next 2-5 words after brand
            after_text = text[match.end():].strip()
            words = re.split(r'\s+', after_text)[:5]  # Get up to 5 words
            
            # Filter out generic terms and units
            filtered = []
            generic_terms = {'supplement', 'dietary', 'natural', 'organic', 
                            'mg', 'ml', 'g', 'oz', 'veggie', 'capsules', 'vegetarian'}
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word and clean_word.lower() not in generic_terms:
                    filtered.append(clean_word)
                if len(filtered) >= 2:  # Minimum 2 words
                    break
                    
            return " ".join(filtered) if filtered else None
        except:
            return None
    
    def extract_title_case_phrases(self, text: str) -> List[str]:
        """Extract 2-4 word phrases in Title Case format"""
        # Find word sequences: Capital + lowercase letters
        patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',         # 2 words
            r'\b([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)\b',  # 3 words
            r'\b([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)\b'  # 4 words
        ]
        
        phrases = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Skip common product terms
                if not any(term in match.lower() for term in ['supplement', 'dietary']):
                    phrases.add(match)
        return list(phrases)
    
    def calculate_overall_confidence(self, brand_confidence: float, product_confidence: float, 
                                   text_quality: float) -> float:
        """Calculate weighted confidence score"""
        # Weights for different factors
        brand_weight = 0.4
        product_weight = 0.3
        text_quality_weight = 0.3
        
        overall = (brand_confidence * brand_weight + 
                  product_confidence * product_weight + 
                  text_quality * text_quality_weight)
        
        return min(overall, 1.0)
    
    def assess_text_quality(self, text: str) -> float:
        """Assess the quality of extracted text"""
        if not text:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length factor
        if len(text) > 10:
            score += 0.2
        if len(text) > 50:
            score += 0.1
        
        # Word ratio (more words = potentially better)
        words = text.split()
        if len(words) > 3:
            score += 0.1
        
        # Capitalization (brands are often capitalized)
        capitalized_words = sum(1 for word in words if word[0].isupper())
        if capitalized_words > 0:
            score += min(capitalized_words * 0.05, 0.2)
        
        return min(score, 1.0)

# Initialize processors
ocr_processor = EnhancedOCRProcessor()
nlp_parser = EnhancedNLPParser()

@app.post("/api/detect", response_model=DetectionResponse)
async def detect_brand_and_product(
    image: UploadFile = File(...),
    brands: str = Form(...),  # JSON string for Vercel compatibility
):
    try:
        # Parse JSON string for brand info
        brand_data_list = json.loads(brands)
        brand_infos = [BrandInfo(**data) for data in brand_data_list]
        
        # Read image
        image_bytes = await image.read()
        
        # Extract text using enhanced OCR
        extracted_text = await ocr_processor.extract_text(image_bytes)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the image")
        
        # Assess text quality
        text_quality = nlp_parser.assess_text_quality(extracted_text)
        
        # Try to match each brand
        best_match = None
        for brand_info in brand_infos:
            brand_matches = nlp_parser.extract_brand_names(extracted_text, brand_info)
            if brand_matches:
                if not best_match or brand_matches[0]['confidence'] > best_match['confidence']:
                    best_match = {
                        'brand_info': brand_info,
                        'match': brand_matches[0]
                    }
        
        # Extract product information using OpenAI
        best_brand = best_match['match'] if best_match else None
        product_info = await nlp_parser.extract_product_with_openai(
            extracted_text, 
            best_match['brand_info'] if best_match else brand_infos[0]
        )
        
        # Calculate overall confidence
        brand_conf = best_brand['confidence'] if best_brand else 0.0
        product_conf = product_info['product_confidence']
        
        overall_confidence = nlp_parser.calculate_overall_confidence(
            brand_conf, product_conf, text_quality
        )

        # Generate renamed file information
        brand_name = best_brand['brand'] if best_brand else "Unknown"
        product_name = product_info['best_product'] or "UnknownProduct"
        
        # Clean the names for file system use
        clean_brand_name = re.sub(r'[^a-zA-Z0-9]', '_', brand_name)
        clean_product_name = re.sub(r'[^a-zA-Z0-9]', '_', product_name)
        
        # Get original file extension
        original_filename = image.filename
        file_extension = os.path.splitext(original_filename)[1]
        
        # Create new filename
        new_filename = f"{clean_product_name}_{clean_brand_name}{file_extension}"
        
        return DetectionResponse(
            brand=brand_name,
            best_product=product_name,
            confidence=overall_confidence,
            extracted_text=extracted_text,
            processing_metadata={
                'text_quality': text_quality,
                'brand_confidence': brand_conf,
                'product_confidence': product_conf,
                'total_brands_found': len(brand_infos),
                'ocr_method': 'google_vision' if GOOGLE_VISION_AVAILABLE else 'tesseract',
                'product_detection_method': 'openai , google vision',
                'explanation': product_info.get('explanation', '')
            },
            renamed_file={
                'original_name': original_filename,
                'new_name': new_filename
            }
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in form data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for Vercel"""
    return {
        "status": "healthy",
        "ocr_available": GOOGLE_VISION_AVAILABLE,
        "version": "2.0.0"
    }

# Vercel handler
handler = app