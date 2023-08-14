from transformers import ViTImageProcessor, ViTForImageClassification

MODEL_NAME_OR_PATH = "google/vit-base-patch16-224"

def download_model() -> tuple:
    """Download the model and processor."""
    model = ViTForImageClassification.from_pretrained(MODEL_NAME_OR_PATH)
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)
    return model, processor

if __name__ == "__main__":
    download_model()
    