#!/usr/bin/env python
"""Quick test of the ONNX model loading with GreenOnGreen class"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
from utils.greenongreen import GreenOnGreen

def test_onnx_model():
    print("Testing ONNX model loading...")
    
    # Initialize the detector
    try:
        detector = GreenOnGreen(model_path="models")
        print(f"✓ Model loaded successfully: {detector.model_path}")
        print(f"✓ Backend: {detector.backend}")
        print(f"✓ Input size: {detector.inference_size}")
        print(f"✓ Input name: {detector.input_name}")
        print(f"✓ Labels loaded: {len(detector.labels)} classes")
        
        # Create a dummy image for testing
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test inference
        print("\nTesting inference...")
        cnts, boxes, centers, output_img = detector.inference(test_image, confidence=0.5)
        
        print(f"✓ Inference completed")
        print(f"✓ Detected {len(boxes)} objects")
        print(f"✓ Centers: {centers}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_onnx_model()
    if success:
        print("\n🎉 Model test PASSED - ready for Raspberry Pi!")
    else:
        print("\n❌ Model test FAILED")
