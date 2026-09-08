# Ai Fusion Camera V3

Android realtime AI camera/object tracking app.

## V3 goals
- Realtime camera object detection using ML Kit.
- Automatic performance profile for low, mid-range and flagship devices.
- Smaller 2dp detection boxes rather than oversized overlays.
- Full sensor orientation: portrait 9:16 and landscape 16:9.
- File picker for local images/videos/documents.
- USB OTG/pendrive access through Android Storage Access Framework when the device exposes the drive.
- Bluetooth settings shortcut.
- AI confidence control.
- Single GitHub Actions workflow builds both debug and release APKs and uploads them as one artifact.

## Build
The repository uses Gradle 8.10.2, JDK 17 and Android Gradle Plugin 8.7.3. Run the **Build Ai Fusion Camera APK** workflow manually from Actions, or push to `main`.

## Important
AI quality is intentionally adaptive: flagship devices receive higher analysis resolution, while lower tiers use lighter analysis to protect realtime performance, battery and thermals. Device-tier detection is a heuristic, not a benchmark.
