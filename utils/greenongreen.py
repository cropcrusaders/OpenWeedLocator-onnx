#!/usr/bin/env python
"""Green-on-Green weed detection.

This module now supports running either TensorFlow Lite models on the
Google Coral or ONNX models via the ONNX Runtime.  When a directory is
provided it will attempt to load the first ``.onnx`` or ``.tflite`` file
alphabetically.

The ONNX branch is designed to run on the Raspberry Pi 5 without
additional accelerators.
"""

from pathlib import Path

import cv2
import numpy as np
try:
    import onnxruntime as ort  # Required when using ONNX models
except Exception:
    # Allow import to succeed on systems that only use TFLite/Coral
    ort = None


class GreenOnGreen:
    def __init__(self, model_path="models", label_file="models/labels.txt"):
        if model_path is None:
            print(
                "[WARNING] No model directory or path provided with --model-path "
                "flag. Attempting to load from default..."
            )
            model_path = "models"

        self.model_path = Path(model_path)

        if self.model_path.is_dir():
            model_files = sorted(list(self.model_path.glob("*.onnx")))
            if not model_files:
                raise FileNotFoundError(
                    "No .onnx model files found. Please provide a directory "
                    "containing .onnx files or specify an .onnx model file directly."
                )
            self.model_path = model_files[0]
            print(f"[INFO] Using {self.model_path.stem} model...")

        elif self.model_path.suffix == ".onnx":
            print(f"[INFO] Using {self.model_path.stem} model...")

        else:
            print(
                f"[WARNING] Specified model path {model_path} is unsupported, "
                "attempting to use default..."
            )
            model_files = sorted(list(Path("models").glob("*.onnx")))
            try:
                self.model_path = model_files[0]
                print(f"[INFO] Using {self.model_path.stem} model...")
            except IndexError:
                raise FileNotFoundError("No .onnx model files found.") from None

        self.labels = self._read_label_file(label_file)
        self.backend = None

        if self.model_path.suffix == ".onnx":

            if ort is None:
                raise ImportError(
                    "onnxruntime is not installed. Please install it for ONNX models."
                )

            self.backend = "onnx"
            self.session = ort.InferenceSession(
                self.model_path.as_posix(), providers=["CPUExecutionProvider"]
            )
            input_shape = self.session.get_inputs()[0].shape
            # input shape expected as [batch, channels, height, width]
            self.input_name = self.session.get_inputs()[0].name
            self.inference_size = (input_shape[3], input_shape[2])

        elif self.model_path.suffix == ".tflite":
            raise ImportError(
                "TensorFlow Lite/Coral support has been removed. "
                "Please use ONNX models (.onnx files) instead. "
                "Convert your model to ONNX format: yolo export model=best.pt format=onnx"
            )

        else:
            raise ValueError(
                "Unsupported model format. Expected .onnx file only. "
                "TensorFlow Lite support has been removed."
            )

    @staticmethod
    def _read_label_file(label_file):
        labels = {}
        path = Path(label_file)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    pair = line.strip().split(maxsplit=1)
                    if len(pair) == 2 and pair[0].isdigit():
                        labels[int(pair[0])] = pair[1]
        return labels

    def inference(self, image, confidence=0.5, filter_id=None):
        height, width, _ = image.shape
        self.boxes = []
        self.weed_centers = []

        # ONNX inference only
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.inference_size)
        img_norm = img_resized.astype(np.float32) / 255.0
        img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :]

        detections = self.session.run(None, {self.input_name: img_input})[0][0]

        for det in detections:
            if len(det) < 5:
                # Skip detections with unexpected format
                continue
            x, y, w, h, obj_conf, *class_conf = det
            if len(class_conf) == 0:
                class_conf = [1.0]
            class_id = int(np.argmax(class_conf))
            score = obj_conf * class_conf[class_id]
            if score < confidence:
                continue
            if filter_id is not None and class_id != filter_id:
                continue

            x1 = int((x - w / 2) * width / self.inference_size[0])
            y1 = int((y - h / 2) * height / self.inference_size[1])
            x2 = int((x + w / 2) * width / self.inference_size[0])
            y2 = int((y + h / 2) * height / self.inference_size[1])
            box_w = x2 - x1
            box_h = y2 - y1

            self.boxes.append([x1, y1, box_w, box_h])
            center_x = int(x1 + box_w / 2)
            center_y = int(y1 + box_h / 2)
            self.weed_centers.append([center_x, center_y])

            percent = int(score * 100)
            label = self.labels.get(class_id, str(class_id))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                image,
                f"{percent}% {label}",
                (x1, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )

        return None, self.boxes, self.weed_centers, image

