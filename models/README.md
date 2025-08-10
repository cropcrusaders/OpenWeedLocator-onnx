# Model Setup for OpenWeedLocator ONNX

This version of OpenWeedLocator uses **ONNX models only** for weed detection on Raspberry Pi 5 and other systems without requiring Google Coral accelerators.

## Quick Setup

### 1. Install ONNX Runtime

```bash
pip install onnxruntime
```

### 2. Add Your Model

Place your `.onnx` model file in this directory (`models/`). The system will automatically detect and use the first `.onnx` file it finds alphabetically.

### 3. Configure

Update your configuration file (`config/ONNX_GOG.ini`):
- Set `model_path = models` to auto-detect your ONNX file
- Or set `model_path = models/your_model.onnx` for a specific model

### 4. Labels

Ensure `models/labels.txt` matches your model's classes:
```
0 weed
1 crop
2 other_class
```

## Model Training and Export

### Recommended: YOLOv8 → ONNX

1. **Train your model:**
```bash
yolo train data=your_dataset.yaml model=yolov8n.pt epochs=100
```

2. **Export to ONNX:**
```bash
yolo export model=best.pt format=onnx
```

3. **Copy to models directory:**
```bash
cp best.onnx ~/owl/models/weed_model.onnx
```

## Testing Your ONNX Setup

### Step 1: Test ONNX Runtime Installation

Navigate to your owl directory and test the ONNX Runtime installation:

```bash
owl@raspberrypi:~ $ cd ~/owl
owl@raspberrypi:~/owl $ workon owl
(owl) owl@raspberrypi:~/owl $ python -c "import onnxruntime as ort; print('ONNX Runtime version:', ort.__version__)"
```

If this runs successfully, ONNX Runtime is properly installed.

### Step 2: Test with Your Model

Place your `.onnx` model file in the `models` directory and test it:

```bash
(owl) owl@raspberrypi:~/owl $ python owl.py --show-display --algorithm gog
```

**NOTE**: If you are testing indoors, the camera settings may be too dark. You may need to adjust exposure:

```bash
(owl) owl@raspberrypi:~/owl $ python owl.py --show-display --algorithm gog --exp-compensation 4
```

If this runs correctly, you should see a video feed with detection boxes around objects classified by your weed detection model.

## Advanced: Training Your Own Models

For the best results, you'll want to train models specifically for your weeds and crops. The ONNX approach supports any model architecture that can be exported to ONNX format.

### Recommended Workflow

1. **Collect Data**: Use OWL's data collection mode to gather images
2. **Annotate**: Use tools like [Roboflow](https://roboflow.com/) or [CVAT](https://www.cvat.ai/)
3. **Train**: Use YOLOv8 or YOLOv5 for best results
4. **Export**: Convert trained model to ONNX format
5. **Deploy**: Place ONNX model in the `models` directory

### YOLOv8 Training Example

```bash
# Install YOLOv8
pip install ultralytics

# Train model
yolo train data=your_dataset.yaml model=yolov8n.pt epochs=100

# Export to ONNX
yolo export model=runs/detect/train/weights/best.pt format=onnx

# Copy to OWL
cp runs/detect/train/weights/best.onnx ~/owl/models/weed_model.onnx
```

### Data Collection with OWL

You can use OWL itself to collect training data by enabling image sampling in your config file:

```ini
[DataCollection]
sample_images = True
sample_method = whole
sample_frequency = 30
```

This will automatically save images to help build your training dataset.

Currently, the `GreenOnGreen` class will load the first (alphabetically) ONNX model in the directory if specified with
`algorithm='gog'` or will load a specific model if `algorithm=path/to/model.onnx`. Ensure all your classes
appear in the `labels.txt` file in the correct order.

## References

These are some of the sources used in the development of this ONNX-based approach:

1. [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
2. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
3. [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
4. [Weed-AI Dataset Collection](https://weed-ai.sydney.edu.au/)
5. [Roboflow Annotation Platform](https://roboflow.com/)
