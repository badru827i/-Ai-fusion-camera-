# Ai Fusion Camera V3

Android realtime AI camera/object tracking app.

## AI engine
- Realtime camera inference currently uses Google ML Kit as the safe fallback runtime.
- The project is prepared for **YOLO11x-Pose** integration.
- The supplied YOLO11x-Pose checkpoint is a PyTorch/Ultralytics `.pt` model. Android cannot execute that raw checkpoint directly.
- The production path is to convert it to **ONNX** or **TensorFlow Lite**, then run it on-device with an Android runtime.
- The model configuration is stored in `models/model-config.json`.
- The raw 118 MB checkpoint should not be bundled into the APK.

## V3 goals
- Realtime camera object detection using ML Kit fallback.
- YOLO11x-Pose runtime integration after model conversion.
- Automatic performance profile for low, mid-range and flagship devices.
- Smaller detection boxes rather than oversized overlays.
- Full sensor orientation: portrait 9:16 and landscape 16:9.
- File picker for local images/videos/documents.
- USB OTG/pendrive access through Android Storage Access Framework when the device exposes the drive.
- Bluetooth settings shortcut.
- AI confidence, FPS, object-limit and persistence controls.
- Saved detection history with a bounded local database-like store.
- Single GitHub Actions workflow builds debug and release APKs.

## Build
The repository uses Gradle 8.10.2, JDK 17 and Android Gradle Plugin 8.7.3. Run the **Build Ai Fusion Camera APK** workflow manually from Actions, or push to `main`.

## Important
AI quality is intentionally adaptive: flagship devices receive higher analysis resolution, while lower tiers use lighter analysis to protect realtime performance, battery and thermals. Device-tier detection is a heuristic, not a benchmark.

See `models/README.md` for the YOLO11x-Pose deployment requirements.
