# Pakistani Number Plate Detector

A production-ready object detection API that detects Pakistani number plates in images using a fine-tuned YOLOv8 model, served via FastAPI and containerized with Docker.

## Results

| Metric | Score |
|--------|-------|
| mAP50 | 97.5% |
| Precision | 100% |
| Recall | 96.8% |
| Inference Speed | 2.5ms/image |

## Tech Stack

- **Model:** YOLOv8n fine-tuned on Pakistani number plates
- **Export:** ONNX for framework-independent inference
- **API:** FastAPI with NMS postprocessing
- **Containerization:** Docker

## Project Structure

```
number-plate-detector/
├── app.py           # FastAPI application
├── best.onnx        # Trained ONNX model
├── Dockerfile       # Container definition
├── requirements.txt # Dependencies
└── README.md
```

## How It Works

1. Client sends a POST request with an image
2. Image is preprocessed to 640x640 and normalized
3. ONNX Runtime runs inference on the image
4. Non-Maximum Suppression removes duplicate detections
5. Clean JSON response with bounding boxes and confidence scores

## API Usage

### Run with Docker

```bash
docker run -p 8000:8000 number-plate-detector
```

### Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

### Endpoint

```
POST /detect
```

**Request:** Multipart form with image file

**Response:**

```json
{
  "filename": "car.jpg",
  "detections_found": 1,
  "detections": [
    {
      "confidence": 0.88,
      "bbox": {
        "x1": 326,
        "y1": 424,
        "x2": 428,
        "y2": 500
      }
    }
  ]
}
```

## Interactive Docs

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI to test the API directly in your browser.

## Dataset

Trained on the [Pakistani Number Plates dataset](https://universe.roboflow.com/detection-and/pakistani-number-plates-el5hd/dataset/2) from Roboflow Universe.

- Train: 235 images
- Validation: 67 images
- Test: 34 images
