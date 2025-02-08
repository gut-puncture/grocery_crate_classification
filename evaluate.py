import os
import base64
import cv2
import numpy as np
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)

# Ensure the GEMINI_API_KEY is set in your environment or HuggingFace Secrets.
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    logging.error("GEMINI_API_KEY not set in environment variables.")
    raise Exception("GEMINI_API_KEY not set in environment variables.")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Heuristic thresholds for image quality (adjust as needed)
BRIGHTNESS_THRESHOLD = 50   # Average brightness (0-255 scale)
BLUR_THRESHOLD = 100        # Variance of Laplacian for blur detection

def is_image_too_dark(image: np.ndarray) -> bool:
    """Return True if the image's average brightness is below the threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    if mean_val < BRIGHTNESS_THRESHOLD:
        logging.info(f"Image brightness too low: {mean_val}")
        return True
    return False

def is_image_blurry(image: np.ndarray) -> bool:
    """Return True if the image is detected to be too blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        logging.info(f"Image is blurry: Variance = {laplacian_var}")
        return True
    return False

def construct_prompt(quality: str) -> str:
    """
    Build the Gemini prompt for evaluating the crate image.
    quality: either "average" or "premium"
    """
    quality = quality.lower()
    if quality not in ["average", "premium"]:
        raise ValueError("quality must be either 'average' or 'premium'")
    prompt = f"""
You are provided with an image of a crate containing a homogeneous type of produce. Evaluate the overall visual quality of the produce items only. Do not consider the crate's arrangement. Use your intuitive judgment to decide whether the produce is excellent or simply acceptable based on the following quality framework.

Quality Framework:

1. Color & Appearance:
   - Premium: The produce exhibits mostly vibrant and consistent color with minimal discoloration, dark spots, or blemishes, indicating an excellent appearance.
   - Average: The produce generally has consistent color with some minor variations or slight discoloration, but overall it is acceptable.
   - Unacceptable: Noticeable discoloration, significant bruising, or prominent dark spots are present.

2. Texture & Surface Condition:
   - Premium: The surface is largely smooth and free of significant blemishes, scratches, or signs of bruising, reflecting a high standard.
   - Average: Minor surface imperfections are acceptable as long as they do not dominate the overall look.
   - Unacceptable: Significant bruising, scars, or rough textures are evident.

3. Shape & Size Consistency:
   - Premium: Items are mostly uniform in shape and size, conveying a high-quality standard.
   - Average: Items show acceptable consistency with some natural variations.
   - Unacceptable: Noticeable irregularities, misshapen items, or uneven sizes indicate lower quality.

4. Freshness & Condition:
   - Premium: Each produce item appears fresh and vibrant, with little to no signs of overripeness, wilting, or decay.
   - Average: Produce items are generally fresh, though there might be minor signs of aging that do not affect overall quality.
   - Unacceptable: Items appear significantly overripe, wilted, or visibly damaged.

For a crate to be approved for a premium customer, the overall quality of the produce should be excellent – it should meet the Premium criteria in most aspects, even if minor imperfections exist.
For an average customer, the produce should be acceptable overall – meeting at least the Average criteria in all aspects, with any minor issues not being a major concern.

Evaluate the crate image using your intuition based on the above framework.
The standard to apply is: acceptable for {quality} customer.

Respond with only one word: either "approved" if the produce meets the specified standard, or "not approved" if it does not. If the image is too dark, too blurry, or insufficient to evaluate, respond with "image not good enough for evaluation".
    """
    return prompt.strip()

def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to a Base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def evaluate_image(file_bytes: bytes, quality: str) -> str:
    """
    Evaluate the produce quality in the image.
    
    Args:
      file_bytes: The raw bytes of the uploaded image.
      quality: "average" or "premium" (determines the quality threshold).
      
    Returns:
      A single word string: "approved", "not approved", or "image not good enough for evaluation".
    """
    # Decode the image from bytes
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        logging.error("Failed to decode image.")
        return "image not good enough for evaluation"
    
    # Check if image quality is sufficient
    if is_image_too_dark(image) or is_image_blurry(image):
        return "image not good enough for evaluation"
    
    # Build the prompt based on the selected quality threshold
    try:
        prompt = construct_prompt(quality)
    except ValueError as ve:
        logging.error(f"Invalid quality parameter: {ve}")
        return "not approved"
    
    # Encode the image to Base64 for the Gemini API call
    image_b64 = encode_image_to_base64(file_bytes)
    mime_type = "image/jpeg"  # Assuming JPEG for simplicity
    
    inputs = [
        {
            "mime_type": mime_type,
            "data": image_b64
        },
        prompt
    ]
    
    try:
        response = model.generate_content(inputs)
        output_text = response.text.strip().lower()
        logging.info(f"Gemini response: {output_text}")
        if output_text not in ["approved", "not approved", "image not good enough for evaluation"]:
            output_text = "not approved"
        return output_text
    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}")
        return "not approved"
