# AI Model Integration

## YOLO11x-Pose supplied by the project owner

The Android app is prepared to use a YOLO pose model, but the supplied `yolo11x-pose.pt` is a PyTorch/Ultralytics checkpoint. A raw `.pt` checkpoint is **not directly executable by Android**.

### Required runtime artifact

Convert the checkpoint to one of these Android-friendly formats:

- **ONNX** — recommended for an ONNX Runtime based detector.
- **TensorFlow Lite (`.tflite`)** — recommended for a fully Android-native deployment.

The conversion must preserve the YOLO11 pose output (person/object boxes plus 17-keypoint pose data where applicable).

### Important

Do not put the 118 MB raw checkpoint inside the APK. It would make the application unnecessarily large and slow to install/update. Store the converted, optimized runtime model as a release asset or download it on first run with checksum verification.

### Current fallback

Until the runtime model is converted, the app continues to use its existing ML Kit realtime object detector so the camera remains functional.
