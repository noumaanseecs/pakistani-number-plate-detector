from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
import cv2

# Initialize FastAPI app
app = FastAPI(title="Pakistani Number Plate Detector")

# Load ONNX model once at startup
# We load it once so every request doesn't reload the model
session = ort.InferenceSession(
    'best.onnx',
    providers=['CPUExecutionProvider']
)

input_name = session.get_inputs()[0].name


def preprocess(image_bytes):
    # Convert uploaded bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Resize to 640x640 (what model expects)
    img_resized = cv2.resize(img, (640, 640))

    # Normalize to 0-1 and convert HWC to CHW format
    img_normalized = img_resized / 255.0
    img_transposed = img_normalized.transpose(2, 0, 1)

    # Add batch dimension → shape becomes (1, 3, 640, 640)
    return np.expand_dims(img_transposed, axis=0).astype(np.float32)


def postprocess(outputs, conf_threshold=0.5, iou_threshold=0.45):
    predictions = outputs[0][0]
    predictions = predictions.T  # shape: (8400, 5)

    boxes = []
    confidences = []

    for pred in predictions:
        confidence = pred[4]
        if confidence > conf_threshold:
            x_center, y_center, width, height = pred[0], pred[1], pred[2], pred[3]
            # Convert to x1,y1,x2,y2 format for NMS
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))

    if not boxes:
        return []

    # Apply NMS — removes duplicate overlapping boxes
    import cv2
    boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, conf_threshold, iou_threshold)

    detections = []
    for i in indices:
        b = boxes[i]
        detections.append({
            "confidence": round(confidences[i], 2),
            "bbox": {
                "x1": b[0], "y1": b[1],
                "x2": b[2], "y2": b[3]
            }
        })

    return detections


@app.get("/")
def home():
    return {"message": "Pakistani Number Plate Detector is running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()

    # Preprocess
    input_tensor = preprocess(image_bytes)

    # Run inference
    outputs = session.run(None, {input_name: input_tensor})

    # Postprocess
    detections = postprocess(outputs)

    return JSONResponse({
        "filename": file.filename,
        "detections_found": len(detections),
        "detections": detections
    })