# Grocery Crate Quality Evaluator

This repository provides a complete API tool to evaluate the quality of produce in a crate using Google's Gemini multi-modal model (**gemini-2.0-flash-exp**). The tool is designed for a warehouse environment where a packer can simply take a photo of a crate containing homogeneous produce (e.g., all apples or all oranges) and receive a binary "approved" or "not approved" decision based solely on the visual quality of the produce items.
check out the demo here: https://huggingface.co/spaces/AngryYoungBhaloo/grocery_crate_classification

## Features

- **Image Quality Checks:** Automatically rejects images that are too dark or too blurry.
- **Intuitive Evaluation:** Uses Gemini's vision capabilities to provide a binary decision based on a detailed quality framework.
- **Quality Framework:** Evaluates produce on:
  - **Color & Appearance**
  - **Texture & Surface Condition**
  - **Shape & Size Consistency**
  - **Freshness & Condition**
- **Configurable Quality Standard:** Supports two static quality thresholds:
  - **Average:** Acceptable for an average customer (minor imperfections tolerated).
  - **Premium:** Must be nearly flawless, acceptable only for premium customers.
- **Simple API Integration:** No UI is built in; the service is provided via a RESTful API for integration with warehouse or inventory management systems.

## Prerequisites

- Python 3.8+
- A valid Google Gemini API key. Set it in your environment variable `GEMINI_API_KEY`.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gut-puncture/grocery_crate_classification.git
