import os
import base64
import cv2
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS if needed (here we allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    logging.error("GEMINI_API_KEY not set in environment variables.")
    raise Exception("GEMINI_API_KEY not set in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# Generation configuration for gemini-2.0-flash-exp
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

# Image quality thresholds (these are heuristic values)
BRIGHTNESS_THRESHOLD = 50   # Average brightness threshold (0-255 scale)
BLUR_THRESHOLD = 100        # Variance of Laplacian threshold for blur detection

def is_image_too_dark(image: np.ndarray) -> bool:
    """
    Convert image to grayscale and compute average brightness.
    Return True if the image is too dark.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    if mean_val < BRIGHTNESS_THRESHOLD:
        logging.info(f"Image brightness too low: {mean_val}")
        return True
    return False

def is_image_blurry(image: np.ndarray) -> bool:
    """
    Use the variance of the Laplacian to detect blur.
    Return True if the image is too blurry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        logging.info(f"Image blur detected: Variance = {laplacian_var}")
        return True
    return False

def construct_prompt(quality_level: str) -> str:
    """
    Construct the Gemini prompt based on the quality level.
    quality_level: "average" or "premium"
    """
    quality_level = quality_level.lower()
    if quality_level not in ["average", "premium"]:
        raise ValueError("quality_level must be either 'average' or 'premium'")

    prompt = f"""
You are provided with an image of a crate containing a homogeneous type of produce. Evaluate the overall visual quality of the produce items only, ignoring any crate arrangement. Do not perform any item-by-item quantitative analysis; instead, use your intuitive judgment to assess the overall quality.

Quality Framework:

1. Color & Appearance:
   - Premium: The produce exhibits a uniformly vibrant and consistent color with no discoloration, dark spots, or blemishes.
   - Average: The produce shows generally consistent color with minor variations or slight discoloration.
   - Unacceptable: Noticeable discoloration, bruising, or dark spots are present.

2. Texture & Surface Condition:
   - Premium: The surface is smooth and free of any visible blemishes, scratches, or signs of bruising.
   - Average: Minor surface imperfections are acceptable, but they should not be prominent.
   - Unacceptable: Significant bruising, scars, or rough textures are visible.

3. Shape & Size Consistency:
   - Premium: Items are uniform in shape and size, indicating optimal quality.
   - Average: Items are generally consistent with only slight natural variations.
   - Unacceptable: Noticeable irregularities, misshapen items, or uneven sizes are evident.

4. Freshness & Condition:
   - Premium: Each produce item appears fresh with no signs of overripeness, wilting, or decay.
   - Average: Produce items are generally fresh, though minor signs of aging may be visible.
   - Unacceptable: Items appear overripe, wilted, or otherwise damaged.

For an "acceptable for average customer" standard, the overall quality of the produce must meet at least the Average criteria in all categories with no Unacceptable issues.
For an "acceptable for premium customer" standard, the overall quality must meet the Premium criteria in all categories.

Evaluate the crate image using your intuition based on the above framework.
The standard to apply is: acceptable for {quality_level} customer.

Respond with only one word: either "approved" if the produce meets the specified standard, or "not approved" if it does not. If the image is too dark, too blurry, or insufficient to evaluate, respond with "image not good enough for evaluation".
    """
    return prompt.strip()

def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to a base64 string.
    """
    return base64.b64encode(image_bytes).decode('utf-8')

@app.post("/evaluate-crate")
async def evaluate_crate(file: UploadFile = File(...), quality: str = Form(...)):
    """
    API endpoint to evaluate the crate image.
    Accepts:
      - file: An image file of the produce crate.
      - quality: A string ("average" or "premium") indicating the quality threshold.
    
    Returns a JSON response with the key "result" whose value is one of:
      - "approved"
      - "not approved"
      - "image not good enough for evaluation" (if the image fails quality checks)
    """
    try:
        # Read uploaded image file into bytes
        file_bytes = await file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            logging.error("Failed to decode image.")
            raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        logging.error(f"Error reading image file: {e}")
        raise HTTPException(status_code=400, detail="Error processing image file.")

    # Run image quality checks
    if is_image_too_dark(image):
        return JSONResponse(status_code=400, content={"result": "image not good enough for evaluation"})
    if is_image_blurry(image):
        return JSONResponse(status_code=400, content={"result": "image not good enough for evaluation"})

    # Build the prompt for the Gemini model based on the quality threshold
    try:
        prompt = construct_prompt(quality)
    except ValueError as ve:
        logging.error(f"Invalid quality parameter: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    # Encode the image to base64 for the Gemini API call
    image_base64 = encode_image_to_base64(file_bytes)
    # Determine the MIME type from the uploaded file; default to image/jpeg if not provided.
    mime_type = file.content_type if file.content_type else "image/jpeg"

    # Prepare the input list for the Gemini API call: image data and prompt text.
    try:
        inputs = [
            {
                "mime_type": mime_type,
                "data": image_base64
            },
            prompt
        ]
        # Call the Gemini API using generate_content with the image and the prompt
        response = model.generate_content(inputs)
        output_text = response.text.strip().lower()
        logging.info(f"Gemini response: {output_text}")

        # Ensure the output is one of the expected responses; if not, default to "not approved"
        if output_text not in ["approved", "not approved", "image not good enough for evaluation"]:
            output_text = "not approved"
    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}")
        raise HTTPException(status_code=500, detail="Error during evaluation process.")

    return JSONResponse(content={"result": output_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
