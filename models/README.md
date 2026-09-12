# AI Model Integration

## YOLO11x-Pose

The Android app now contains a TensorFlow Lite runtime bridge for the supplied YOLO11x-Pose checkpoint. Android does **not** execute the raw PyTorch `.pt` file directly; the checkpoint must first be exported to a compatible `.tflite` deployment model. Ultralytics documents LiteRT/TFLite export for pose models at 640x640. citeturn385757search0turn385757search1

### Expected deployment file

`yolo11x-pose.tflite`

Place it in either:

- `app/src/main/assets/yolo11x-pose.tflite`, or
- import it from the app using **Import YOLO**, which copies it to the app-private `models/` directory.

The Android bridge accepts the common raw YOLO pose output layout equivalent to 56 attributes x 8400 candidates for a single COCO person class: box + class score + 17 keypoints x 3 values. Ultralytics pose models expose 17 keypoints for COCO-Pose. citeturn385757search6turn385757search7

### Runtime behavior

1. CameraX sends frames to the analyzer.
2. If `yolo11x-pose.tflite` is available, YOLO11x-Pose becomes the primary offline detector.
3. Detection boxes are rendered in the existing HUD as `PERSON` + confidence.
4. If the model is not installed, the existing ML Kit detector remains active as a fallback so the camera still works.
5. The raw 118 MB `.pt` checkpoint is not packaged into the APK.

### Export example

```bash
yolo export model=yolo11x-pose.pt format=litert imgsz=640 batch=1
```

The resulting model should be named `yolo11x-pose.tflite` and verified before importing into the app. Ultralytics currently recommends LiteRT for on-device deployment and lists 640 as the mobile input size for pose exports. citeturn385757search0
