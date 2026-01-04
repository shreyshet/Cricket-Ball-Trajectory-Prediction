import os
# MUST be set before importing torch/ultralytics
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from ultralytics import YOLO

# Everything that starts a process (like training) MUST be inside this block
if __name__ == '__main__':
    # Load your model
    model = YOLO("yolo11n.pt")

    # Start training
    # Note: Using workers=0 can sometimes bypass this, but it slows training significantly.
    # The proper fix is this 'if __name__ == "__main__":' guard.
    results = model.train(
        data="cricket_data.yaml",
        epochs=15,
        imgsz=640,  # Try 640 first. If stable, try 960. 1280 is likely too big for 6GB.
        batch=4,  # severe reduction to fit in VRAM
        device=0,      # Use 0 for GPU, 'cpu' for CPU
        workers=2  # Reduce data loader workers to save system RAM
    )