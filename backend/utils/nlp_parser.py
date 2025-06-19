import spacy
from typing import List, Dict

nlp = spacy.load("en_core_web_sm")

def parse_brand_and_product(
    text: str, 
    brand_names: List[str], 
    product_types: List[str]
) -> Dict[str, str]:
    """
    Uses NLP to extract brand and product name from text.
    Args:
        text: OCR-extracted text.
        brand_names: List of brands (from frontend).
        product_types: List of product categories (e.g., "Herbal Tea").
    Returns:
        {"brand": str, "product_name": str, "confidence": float}
    """
    doc = nlp(text)
    found_brand = None
    found_product = None

    # Step 1: Check for brand names (case-insensitive)
    for token in doc:
        if token.text.lower() in [b.lower() for b in brand_names]:
            found_brand = token.text
            break

    # Step 2: Find product type (e.g., "Chamomile Tea")
    for phrase in product_types:
        if phrase.lower() in text.lower():
            found_product = phrase
            break

    # Fallback: Use the first 3 words if no product type matched
    if not found_product:
        found_product = " ".join(text.split()[:3])

    confidence = 0.9 if found_brand else 0.5  # Simple confidence scoring

    return {
        "brand": found_brand or "Unknown",
        "product_name": found_product,
        "confidence": confidence
    }