# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import logging
from evaluate import evaluate_image
from google_sheets import update_google_sheet  # New import

import uvicorn

logging.basicConfig(level=logging.INFO)
app = FastAPI()

@app.post("/evaluate-crate")
async def evaluate_crate(file: UploadFile = File(...), quality: str = Form(...)):
    """
    Evaluate the quality of a produce crate image.
    
    Parameters:
      - file: The image file.
      - quality: A string ("average" or "premium").
    
    Returns a JSON response with "result" set to one of:
      - "approved"
      - "not approved"
      - "image not good enough for evaluation"
    """
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading image file.")
    
    result = evaluate_image(file_bytes, quality)
    
    # Update the Google Sheet with the image, threshold, and classification
    try:
        update_google_sheet(file_bytes, quality, result)
    except Exception as e:
        logging.error(f"Error updating Google Sheet: {e}")
    
    return JSONResponse(content={"result": result})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
