package com.badru.aifusioncamera

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { AiFusionCamera() }
    }
}

data class BoxData(val l: Float, val t: Float, val r: Float, val b: Float)

enum class DeviceTier { LOW, MID, FLAGSHIP }

private fun tier(context: Context): DeviceTier {
    val cores = Runtime.getRuntime().availableProcessors()
    val ram = (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).memoryClass / 1024f
    return when {
        cores >= 8 && ram >= 8f -> DeviceTier.FLAGSHIP
        cores >= 6 && ram >= 5f -> DeviceTier.MID
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
        if (granted) CameraPreview(profile, confidence) { boxes = it }
        else Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Camera permission diperlukan", color = Color.White)
            Spacer(Modifier.height(12.dp))
            Button({ cameraPermission.launch(Manifest.permission.CAMERA) }) { Text("Allow Camera") }
        }
        if (granted) BoxOverlay(boxes, Modifier.fillMaxSize())
        Column(Modifier.fillMaxWidth().align(Alignment.TopCenter).padding(14.dp)) {
            Text("AI FUSION CAMERA", color = Color.White, style = MaterialTheme.typography.titleLarge)
            Text("V3  •  ${profile.name}  •  AUTO ROTATION", color = Color.White.copy(.8f))
        }
        Card(Modifier.fillMaxWidth().align(Alignment.BottomCenter).padding(12.dp), shape = RoundedCornerShape(18.dp)) {
            Row(Modifier.fillMaxWidth().padding(8.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
                Button({ files.launch(arrayOf("image/*", "video/*", "application/*", "text/*")) }) { Text("Files") }
                Button({ tree.launch(null) }) { Text("USB / Drive") }
                Button({ settings = true }) { Text("Settings") }
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
        ObjectDetection.getClient(ObjectDetectorOptions.Builder().setDetectorMode(ObjectDetectorOptions.STREAM_MODE).enableMultipleObjects().enableClassification().build())
    }
    DisposableEffect(Unit) { onDispose { detector.close(); executor.shutdown() } }
    AndroidView(Modifier.fillMaxSize(), factory = { ctx ->
        val view = PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.surfaceProvider = view.surfaceProvider }
            val width = if (profile == DeviceTier.FLAGSHIP) 1280 else if (profile == DeviceTier.MID) 960 else 640
            val height = if (profile == DeviceTier.FLAGSHIP) 720 else if (profile == DeviceTier.MID) 540 else 360
            val analysis = ImageAnalysis.Builder().setTargetResolution(android.util.Size(width, height)).setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
            analysis.setAnalyzer(executor) { proxy ->
                val image = proxy.image
                if (image == null) { proxy.close(); return@setAnalyzer }
                detector.process(InputImage.fromMediaImage(image, proxy.imageInfo.rotationDegrees))
                    .addOnSuccessListener { result ->
                        onBoxes(result.mapNotNull { o ->
                            val label = o.labels.maxByOrNull { it.confidence }
                            if (label != null && label.confidence >= confidence) {
                                val r = o.boundingBox
                                BoxData(r.left.toFloat(), r.top.toFloat(), r.right.toFloat(), r.bottom.toFloat())
                            } else null
                        })
                    }.addOnCompleteListener { proxy.close() }
            }
            try { provider.unbindAll(); provider.bindToLifecycle(context as ComponentActivity, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis) } catch (_: Exception) { }
        }, ContextCompat.getMainExecutor(ctx))
        view
    })
}

@Composable
private fun BoxOverlay(boxes: List<BoxData>, modifier: Modifier) {
    Canvas(modifier) {
        val sx = size.width / 1280f
        val sy = size.height / 720f
        boxes.forEach { b ->
            val l = (b.l * sx).coerceIn(0f, size.width)
            val t = (b.t * sy).coerceIn(0f, size.height)
            val r = (b.r * sx).coerceIn(0f, size.width)
            val bot = (b.b * sy).coerceIn(0f, size.height)
            drawRect(Color.Cyan, Offset(l, t), androidx.compose.ui.geometry.Size((r-l).coerceAtLeast(2f), (bot-t).coerceAtLeast(2f)), style = androidx.compose.ui.graphics.drawscope.Stroke(2.dp.toPx()))
        }
    }
}

@Composable
private fun Settings(value: Float, onValue: (Float) -> Unit, close: () -> Unit, bluetooth: () -> Unit) {
    AlertDialog(onDismissRequest = close, title = { Text("Ai Fusion Camera Settings") }, text = {
        Column {
            Text("AI confidence ${(value * 100).toInt()}%")
            Slider(value, onValue, valueRange = .35f..0.9f)
            Text("Orientation: Auto • 16:9 / 9:16")
            Text("Performance: automatic device profile")
            Text("Files: Android Storage Access Framework")
            Text("USB OTG / pendrive: supported when Android exposes the drive")
            Spacer(Modifier.height(10.dp))
            Button(bluetooth, Modifier.fillMaxWidth()) { Text("Connect Bluetooth") }
        }
    }, confirmButton = { TextButton(close) { Text("Done") } })
}
