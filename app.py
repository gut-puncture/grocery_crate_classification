# app.py
import gradio as gr
import cv2
from evaluate import evaluate_image
from google_sheets import update_google_sheet  # New import

def process(image, quality_code):
    """
    Process the uploaded image and quality threshold.
    
    Args:
      image: The image as a numpy array (from Gradio).
      quality_code: "1" for premium, "2" for average.
    
    Returns:
      A string: "approved", "not approved", or an error message.
    """
    # Map numeric input to quality threshold string
    if quality_code == "1":
        quality = "premium"
    elif quality_code == "2":
        quality = "average"
    else:
        return "Invalid quality selection. Please choose 1 for premium or 2 for average."
    
    if image is None:
        return "No image provided."
    
    # Gradio returns the image as an RGB numpy array. Convert to BGR for OpenCV.
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Encode the image to bytes (JPEG)
    success, encoded_image = cv2.imencode('.jpg', image_bgr)
    if not success:
        return "Error encoding image."
    file_bytes = encoded_image.tobytes()
    
    # Evaluate the image using the shared function
    result = evaluate_image(file_bytes, quality)
    
    # Update the Google Sheet with the image, threshold, and classification
    try:
        update_google_sheet(file_bytes, quality, result)
    except Exception as e:
        # Log the error; the evaluation result is still returned to the user.
        print(f"Error updating Google Sheet: {e}")
    
    return result

title = "Grocery Crate Quality Evaluator"
description = """
Upload an image of a produce crate (with homogeneous items) and select the quality threshold.
- **1**: Premium quality (almost flawless produce)
- **2**: Average quality (acceptable for average customer)
The system will evaluate the produce quality and return one of:
**approved**, **not approved**, or **image not good enough for evaluation**.
Note: The image, threshold, and classification are also logged in a Google Sheet.
"""

iface = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(label="Upload Crate Image", type="numpy"),
        gr.Radio(choices=["1", "2"], label="Select Quality Threshold (1: Premium, 2: Average)")
    ],
    outputs="text",
    title=title,
    description=description,
)

if __name__ == "__main__":
    iface.launch()
