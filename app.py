from potassium import Potassium, Request, Response
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import io
import base64

MODEL_NAME_OR_PATH = "google/vit-base-patch16-224"

app = Potassium("vit-base-patch16-224")

@app.init
def init() -> dict:
    """Initialize the application with the model and processor."""
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME_OR_PATH)
    return {
        "model": model,
        "processor": processor
    }
    
@app.handler()
def handler(context: dict, request: Request) -> Response:
    """Handle a request to generate text from the image."""
    model = context.get("model")
    processor = context.get("processor")
    image_b64 = request.json.get("image")
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    output = model.config.id2label[predicted_class_idx]
    return Response(json={"output": output}, status=200)

if __name__ == "__main__":
    app.serve()