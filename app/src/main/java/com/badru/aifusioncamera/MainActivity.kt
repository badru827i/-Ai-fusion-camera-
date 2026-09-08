package com.badru.aifusioncamera

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.RectF
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import java.util.concurrent.Executors
import kotlin.math.max

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { AiFusionCamera() }
    }
}

data class BoxData(
    val rect: RectF,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val rotationDegrees: Int
)

enum class DeviceTier { LOW, MID, FLAGSHIP }

private fun tier(context: Context): DeviceTier {
    val info = ActivityManager.MemoryInfo()
    (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(info)
    val ramGb = info.totalMem / (1024.0 * 1024.0 * 1024.0)
    val cores = Runtime.getRuntime().availableProcessors()
    return when {
        cores >= 8 && ramGb >= 8.0 -> DeviceTier.FLAGSHIP
        cores >= 6 && ramGb >= 5.0 -> DeviceTier.MID
        else -> DeviceTier.LOW
    }
}

@Composable
private fun AiFusionCamera() {
    val context = LocalContext.current
    val profile = remember { tier(context) }
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    var settings by remember { mutableStateOf(false) }
    var confidence by remember { mutableFloatStateOf(.55f) }
    var boxes by remember { mutableStateOf(emptyList<BoxData>()) }
    val cameraPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    val files = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { }
    val tree = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (granted) {
            CameraPreview(profile, confidence) { boxes = it }
            BoxOverlay(boxes, Modifier.fillMaxSize())
        } else {
            Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
                Text("Camera permission diperlukan", color = Color.White)
                Spacer(Modifier.height(12.dp))
                Button(onClick = { cameraPermission.launch(Manifest.permission.CAMERA) }) { Text("Allow Camera") }
            }
        }
        Column(Modifier.fillMaxWidth().align(Alignment.TopCenter).padding(14.dp)) {
            Text("AI FUSION CAMERA", color = Color.White, style = MaterialTheme.typography.titleLarge)
            Text("V3  •  ${profile.name}  •  AUTO ROTATION", color = Color.White.copy(.8f))
        }
        Card(Modifier.fillMaxWidth().align(Alignment.BottomCenter).padding(12.dp), shape = RoundedCornerShape(18.dp)) {
            Row(Modifier.fillMaxWidth().padding(8.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
                Button(onClick = { files.launch(arrayOf("image/*", "video/*", "application/*", "text/*")) }) { Text("Files") }
                Button(onClick = { tree.launch(null) }) { Text("USB / Drive") }
                Button(onClick = { settings = true }) { Text("Settings") }
            }
        }
    }
    if (settings) Settings(confidence, { confidence = it }, { settings = false }) {
        context.startActivity(Intent(android.provider.Settings.ACTION_BLUETOOTH_SETTINGS))
    }
}

@Composable
private fun CameraPreview(profile: DeviceTier, confidence: Float, onBoxes: (List<BoxData>) -> Unit) {
    val context = LocalContext.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    val detector = remember {
        ObjectDetection.getClient(
            ObjectDetectorOptions.Builder()
                .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
                .enableMultipleObjects()
                .enableClassification()
                .build()
        )
    }
    DisposableEffect(Unit) { onDispose { detector.close(); executor.shutdown() } }

    AndroidView(
        modifier = Modifier.fillMaxSize(),
        factory = { ctx ->
            val view = PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
            val future = ProcessCameraProvider.getInstance(ctx)
            future.addListener({
                val provider = future.get()
                val preview = Preview.Builder().build().also { it.surfaceProvider = view.surfaceProvider }
                val size = when (profile) {
                    DeviceTier.FLAGSHIP -> android.util.Size(1280, 720)
                    DeviceTier.MID -> android.util.Size(960, 540)
                    DeviceTier.LOW -> android.util.Size(640, 360)
                }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(size)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(executor) { proxy ->
                    val image = proxy.image
                    if (image == null) {
                        proxy.close()
                        return@setAnalyzer
                    }
                    val rotation = proxy.imageInfo.rotationDegrees
                    detector.process(InputImage.fromMediaImage(image, rotation))
                        .addOnSuccessListener { result ->
                            onBoxes(result.map { o ->
                                val r = o.boundingBox
                                BoxData(RectF(r), image.width, image.height, rotation)
                            })
                        }
                        .addOnCompleteListener { proxy.close() }
                }
                try {
                    provider.unbindAll()
                    provider.bindToLifecycle(context as ComponentActivity, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                } catch (_: Exception) { }
            }, ContextCompat.getMainExecutor(ctx))
            view
        }
    )
}

@Composable
private fun BoxOverlay(boxes: List<BoxData>, modifier: Modifier) {
    Canvas(modifier) {
        boxes.forEach { box ->
            val sourceW = if (box.rotationDegrees % 180 == 0) box.sourceWidth else box.sourceHeight
            val sourceH = if (box.rotationDegrees % 180 == 0) box.sourceHeight else box.sourceWidth
            val scale = max(size.width / sourceW.toFloat(), size.height / sourceH.toFloat())
            val cropX = (sourceW * scale - size.width) / 2f
            val cropY = (sourceH * scale - size.height) / 2f
            val r = box.rect
            drawRect(
                Color.Cyan,
                Offset(r.left * scale - cropX, r.top * scale - cropY),
                androidx.compose.ui.geometry.Size((r.width() * scale).coerceAtLeast(2f), (r.height() * scale).coerceAtLeast(2f)),
                style = Stroke(2.dp.toPx())
            )
        }
    }
}

@Composable
private fun Settings(value: Float, onValue: (Float) -> Unit, close: () -> Unit, bluetooth: () -> Unit) {
    AlertDialog(
        onDismissRequest = close,
        title = { Text("Ai Fusion Camera Settings") },
        text = {
            Column {
                Text("AI confidence ${(value * 100).toInt()}%")
                Slider(value = value, onValueChange = onValue, valueRange = .35f..0.9f)
                Text("Orientation: Auto • 16:9 / 9:16")
                Text("Performance: automatic device profile")
                Text("Files: Android Storage Access Framework")
                Text("USB OTG / pendrive: supported when Android exposes the drive")
                Spacer(Modifier.height(10.dp))
                Button(onClick = bluetooth, modifier = Modifier.fillMaxWidth()) { Text("Connect Bluetooth") }
            }
        },
        confirmButton = { TextButton(onClick = close) { Text("Done") } }
    )
}
