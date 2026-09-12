package com.badru.aifusioncamera

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.RectF
import android.net.Uri
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
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max
import kotlin.math.PI
import androidx.compose.ui.graphics.drawscope.rotate

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { AiFusionCamera() }
    }
}

data class BoxData(
    val rect: RectF, val sourceWidth: Int, val sourceHeight: Int, val rotationDegrees: Int,
    val label: String, val confidence: Float, val angle: Float = 0f,
    val keypoints: FloatArray? = null, val model: String = "AI"
)
data class DetectionStat(val label: String, val count: Int, val confidence: Float)
enum class DeviceTier { LOW, MID, FLAGSHIP }

private fun tier(context: Context): DeviceTier {
    val info = ActivityManager.MemoryInfo()
    (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(info)
    val ramGb = info.totalMem / (1024.0 * 1024.0 * 1024.0)
    val cores = Runtime.getRuntime().availableProcessors()
    return when {
        cores >= 8 && ramGb >= 8 -> DeviceTier.FLAGSHIP
        cores >= 6 && ramGb >= 5 -> DeviceTier.MID
        else -> DeviceTier.LOW
    }
}

@Composable
private fun AiFusionCamera() {
    val context = LocalContext.current
    val profile = remember { tier(context) }
    var cameraGranted by remember { mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    var settingsOpen by remember { mutableStateOf(false) }
    var analyticsOpen by remember { mutableStateOf(true) }
    var confidence by remember { mutableFloatStateOf(.55f) }
    var analysisFps by remember { mutableIntStateOf(if (profile == DeviceTier.LOW) 10 else 15) }
    var maxObjects by remember { mutableIntStateOf(20) }
    var boxScale by remember { mutableFloatStateOf(.72f) }
    var modelGeneration by remember { mutableIntStateOf(0) }
    var yoloReady by remember { mutableStateOf(false) }
    var boxes by remember { mutableStateOf(emptyList<BoxData>()) }
    var graph by remember { mutableStateOf(listOf(0)) }
    var modelText by remember { mutableStateOf("Checking YOLO11x-Pose…") }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { cameraGranted = it }
    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        val fileName = context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
        val lower = fileName?.lowercase() ?: ""
        val targetName = when {
            lower.contains("obb") && lower.endsWith(".tflite") -> YoloObbDetector.MODEL_NAME
            lower.contains("pose") && lower.endsWith(".tflite") -> YoloPoseDetector.MODEL_NAME
            else -> null
        }
        if (targetName == null) {
            modelText = "Import gagal • nama fail mesti mengandungi pose atau obb (.tflite)"
            return@rememberLauncherForActivityResult
        }
        try {
            val dir = File(context.filesDir, "models").apply { mkdirs() }
            val target = File(dir, targetName)
            context.contentResolver.openInputStream(uri)?.use { input -> target.outputStream().use { out -> input.copyTo(out) } }
            modelGeneration++
            modelText = "${if (targetName == YoloObbDetector.MODEL_NAME) "YOLO-OBB" else "YOLO11x-Pose"} imported • ${target.length() / (1024 * 1024)} MB"
        } catch (_: Exception) {
            modelText = "Import gagal"
        }
    }

    LaunchedEffect(boxes.size) { graph = (graph + boxes.size).takeLast(30) }

    val stats = remember(boxes) {
        boxes.groupBy { it.label.uppercase() }
            .map { (label, list) -> DetectionStat(label, list.size, list.map { it.confidence }.average().toFloat()) }
            .sortedByDescending { it.count }
    }
    val total = boxes.size
    val avg = if (boxes.isEmpty()) 0f else boxes.map { it.confidence }.average().toFloat()

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (cameraGranted) {
            key(modelGeneration) {
                CameraPreview(profile, confidence, analysisFps, maxObjects,
                    onStatus = { poseActive, obbActive ->
                        yoloReady = poseActive || obbActive
                        modelText = when {
                            poseActive && obbActive -> "YOLO Pose + OBB • ACTIVE"
                            poseActive -> "YOLO11x-Pose • ACTIVE • 17 keypoints"
                            obbActive -> "YOLO-OBB • ACTIVE • rotated boxes"
                            else -> "ML Kit fallback • YOLO models not loaded"
                        }
                    },
                    onBoxes = { boxes = it })
            }
            BoxOverlay(boxes, Modifier.fillMaxSize(), boxScale)
        } else {
            Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(Icons.Default.CameraAlt, null, tint = Color.Cyan, modifier = Modifier.size(42.dp))
                Spacer(Modifier.height(10.dp))
                Text("Camera permission diperlukan", color = Color.White)
                Spacer(Modifier.height(12.dp))
                Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) { Text("Allow Camera") }
            }
        }

        Surface(
            modifier = Modifier.align(Alignment.TopCenter).padding(12.dp),
            color = Color.Black.copy(alpha = .58f),
            shape = RoundedCornerShape(22.dp), tonalElevation = 8.dp
        ) {
            Row(Modifier.padding(horizontal = 14.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Visibility, null, tint = Color.Cyan, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Column {
                    Text("NEXA VISION", color = Color.White, fontWeight = FontWeight.Bold)
                    Text(modelText, color = Color.White.copy(.62f), style = MaterialTheme.typography.labelSmall)
                }
                Spacer(Modifier.width(10.dp))
                Surface(color = Color(0x3322FFAA), shape = RoundedCornerShape(999.dp)) {
                    Text(if (yoloReady) "YOLO" else "AI", color = Color(0xFF55FFB0), style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
                }
            }
        }

        AnimatedVisibility(
            visible = analyticsOpen,
            enter = fadeIn(spring(stiffness = 550f)),
            exit = fadeOut(),
            modifier = Modifier.align(Alignment.TopEnd).padding(top = 82.dp, end = 10.dp)
        ) { AnalyticsPanel(total, avg, stats, graph) }

        Surface(
            modifier = Modifier.align(Alignment.BottomCenter).padding(10.dp),
            color = Color.Black.copy(alpha = .76f),
            shape = RoundedCornerShape(22.dp), tonalElevation = 10.dp
        ) {
            Row(Modifier.padding(7.dp), verticalAlignment = Alignment.CenterVertically) {
                Surface(color = Color.Cyan.copy(alpha = .10f), shape = RoundedCornerShape(999.dp)) {
                    Text("$total OBJECTS", color = Color.Cyan, fontWeight = FontWeight.Bold, modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp))
                }
                Spacer(Modifier.width(6.dp))
                IconButton(onClick = { analyticsOpen = !analyticsOpen }) { Icon(Icons.Default.Analytics, "Analytics", tint = Color.White) }
                IconButton(onClick = { settingsOpen = true }) { Icon(Icons.Default.Tune, "Settings", tint = Color.White) }
                IconButton(onClick = { importLauncher.launch(arrayOf("application/octet-stream", "*/*")) }) { Icon(Icons.Default.Visibility, "Import YOLO model", tint = Color.Cyan) }
            }
        }
    }

    if (settingsOpen) {
        SettingsSheet(confidence, { confidence = it }, analysisFps, { analysisFps = it.toInt().coerceIn(5, 30) }, maxObjects, { maxObjects = it.toInt().coerceIn(1, 50) }, boxScale, { boxScale = it }, { settingsOpen = false }, modelText)
    }
}

@Composable
private fun AnalyticsPanel(total: Int, avg: Float, stats: List<DetectionStat>, graph: List<Int>) {
    Card(
        modifier = Modifier.widthIn(min = 245.dp, max = 305.dp),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = Color.Black.copy(alpha = .72f))
    ) {
        Column(Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Analytics, null, tint = Color.Cyan, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(7.dp))
                Text("AI DETECTION", color = Color.White, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.height(9.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                StatChip("TOTAL", total.toString())
                StatChip("CONF", "${(avg * 100).toInt()}%")
            }
            Spacer(Modifier.height(10.dp))
            Text("LIVE OBJECT GRAPH", color = Color.White.copy(.56f), style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.height(4.dp))
            Sparkline(graph, Modifier.fillMaxWidth().height(64.dp))
            Spacer(Modifier.height(9.dp))
            Text("DETECTED OBJECTS", color = Color.White.copy(.56f), style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.height(5.dp))
            if (stats.isEmpty()) Text("No object detected", color = Color.White.copy(.48f), style = MaterialTheme.typography.bodySmall)
            else LazyColumn(modifier = Modifier.heightIn(max = 155.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) { items(stats.take(8), key = { it.label }) { ObjectRow(it) } }
        }
    }
}

@Composable
private fun StatChip(title: String, value: String) {
    Surface(color = Color.White.copy(.07f), shape = RoundedCornerShape(12.dp)) {
        Row(Modifier.padding(horizontal = 9.dp, vertical = 7.dp)) {
            Text(title, color = Color.White.copy(.48f), style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.width(5.dp))
            Text(value, color = Color.Cyan, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.labelSmall)
        }
    }
}

@Composable
private fun ObjectRow(stat: DetectionStat) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Surface(color = Color.Cyan.copy(.10f), shape = RoundedCornerShape(9.dp)) {
            Text(stat.label, color = Color.White, style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp))
        }
        Spacer(Modifier.width(6.dp))
        Text("×${stat.count}", color = Color.Cyan, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.labelMedium)
        Spacer(Modifier.weight(1f))
        Text("${(stat.confidence * 100).toInt()}%", color = Color.White.copy(.65f), style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun Sparkline(values: List<Int>, modifier: Modifier) {
    Canvas(modifier) {
        if (values.size < 2) return@Canvas
        val maxValue = (values.maxOrNull() ?: 1).coerceAtLeast(1)
        val step = size.width / (values.size - 1).toFloat()
        for (i in 1 until values.size) {
            val p1 = Offset((i - 1) * step, size.height - (values[i - 1] / maxValue.toFloat()) * size.height)
            val p2 = Offset(i * step, size.height - (values[i] / maxValue.toFloat()) * size.height)
            drawLine(Color.Cyan.copy(.75f), p1, p2, strokeWidth = 3f)
        }
    }
}

@Composable
private fun CameraPreview(profile: DeviceTier, confidence: Float, analysisFps: Int, maxObjects: Int, onStatus: (Boolean, Boolean) -> Unit, onBoxes: (List<BoxData>) -> Unit) {
    val context = LocalContext.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    val pose = remember { YoloPoseDetector.load(context) }
    val obb = remember { YoloObbDetector.load(context) }
    val mlKit = remember {
        ObjectDetection.getClient(ObjectDetectorOptions.Builder().setDetectorMode(ObjectDetectorOptions.STREAM_MODE).enableMultipleObjects().enableClassification().build())
    }
    LaunchedEffect(Unit) { onStatus(pose != null, obb != null) }
    DisposableEffect(Unit) { onDispose { pose?.close(); obb?.close(); mlKit.close(); executor.shutdown() } }

    AndroidView(modifier = Modifier.fillMaxSize(), factory = { ctx ->
        val previewView = PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val target = when (profile) {
                DeviceTier.FLAGSHIP -> android.util.Size(1280, 720)
                DeviceTier.MID -> android.util.Size(960, 540)
                DeviceTier.LOW -> android.util.Size(640, 360)
            }
            val analysis = ImageAnalysis.Builder().setTargetResolution(target).setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
            val lastAt = AtomicLong(0L)
            analysis.setAnalyzer(executor) { proxy ->
                val now = System.currentTimeMillis()
                val previous = lastAt.get()
                val minInterval = (1000L / analysisFps.coerceIn(5, 30)).coerceAtLeast(33L)
                if (now - previous < minInterval || !lastAt.compareAndSet(previous, now)) { proxy.close(); return@setAnalyzer }
                try {
                    if (pose != null || obb != null) {
                        val sourceW = if (proxy.imageInfo.rotationDegrees % 180 == 0) proxy.width else proxy.height
                        val sourceH = if (proxy.imageInfo.rotationDegrees % 180 == 0) proxy.height else proxy.width
                        val combined = ArrayList<BoxData>()
                        pose?.detect(proxy, confidence)?.take(maxObjects)?.forEach { d ->
                            combined += BoxData(RectF(d.rect.left * sourceW, d.rect.top * sourceH, d.rect.right * sourceW, d.rect.bottom * sourceH), sourceW, sourceH, 0, "PERSON", d.confidence, keypoints = d.keypoints, model = "POSE")
                        }
                        obb?.detect(proxy, confidence)?.take(maxObjects)?.forEach { d ->
                            combined += BoxData(RectF(d.rect.left * sourceW, d.rect.top * sourceH, d.rect.right * sourceW, d.rect.bottom * sourceH), sourceW, sourceH, 0, "OBB-${d.classIndex}", d.confidence, angle = d.angle, model = "OBB")
                        }
                        onBoxes(combined.sortedByDescending { it.confidence }.take(maxObjects * 2))
                    } else {
                        val image = proxy.image ?: return@setAnalyzer
                        val rotation = proxy.imageInfo.rotationDegrees
                        mlKit.process(InputImage.fromMediaImage(image, rotation)).addOnSuccessListener { results ->
                            val detected = results.mapNotNull { item ->
                                val best = item.labels.maxByOrNull { it.confidence }
                                val score = best?.confidence ?: 0f
                                if (score < confidence) null else BoxData(RectF(item.boundingBox), image.width, image.height, rotation, best?.text ?: "OBJECT", score, model = "MLKIT")
                            }.sortedByDescending { it.confidence }.take(maxObjects)
                            onBoxes(detected)
                        }.addOnCompleteListener { proxy.close() }
                        return@setAnalyzer
                    }
                } catch (_: Throwable) {
                    onBoxes(emptyList())
                } finally {
                    if (pose != null || obb != null) proxy.close()
                }
            }
            try { provider.unbindAll(); provider.bindToLifecycle(context as ComponentActivity, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis) } catch (_: Exception) { }
        }, ContextCompat.getMainExecutor(ctx))
        previewView
    })
}

@Composable
private fun BoxOverlay(boxes: List<BoxData>, modifier: Modifier, scale: Float) {
    val animatedScale by animateFloatAsState(scale.coerceIn(.30f, 1.2f), spring(dampingRatio = .86f, stiffness = 520f), label = "overlay-scale")
    Canvas(modifier) {
        boxes.forEach { box ->
            val sourceW = if (box.rotationDegrees % 180 == 0) box.sourceWidth else box.sourceHeight
            val sourceH = if (box.rotationDegrees % 180 == 0) box.sourceHeight else box.sourceWidth
            val previewScale = max(size.width / sourceW.toFloat(), size.height / sourceH.toFloat())
            val cropX = (sourceW * previewScale - size.width) / 2f
            val cropY = (sourceH * previewScale - size.height) / 2f
            val rect = box.rect
            val cx = rect.centerX() * previewScale - cropX
            val cy = rect.centerY() * previewScale - cropY
            val width = (rect.width() * previewScale * animatedScale).coerceAtLeast(2f)
            val height = (rect.height() * previewScale * animatedScale).coerceAtLeast(2f)
            val angle = box.angle * 180f / PI.toFloat()
            rotate(degrees = angle, pivot = Offset(cx, cy)) {
                drawRect(Color.Cyan.copy(.92f), Offset(cx - width / 2f, cy - height / 2f), androidx.compose.ui.geometry.Size(width, height), style = Stroke(2.dp.toPx()))
            }
            box.keypoints?.let { kp ->
                for (i in 0 until minOf(17, kp.size / 3)) {
                    if (kp[i * 3 + 2] >= .35f) {
                        val x = kp[i * 3] * 640f * previewScale - cropX
                        val y = kp[i * 3 + 1] * 640f * previewScale - cropY
                        drawCircle(Color.Cyan, radius = 3.5f, center = Offset(x, y))
                    }
                }
            }
            val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply { color = android.graphics.Color.WHITE; textSize = 12.dp.toPx(); typeface = android.graphics.Typeface.create(android.graphics.Typeface.DEFAULT, android.graphics.Typeface.BOLD) }
            drawIntoCanvas { canvas -> canvas.nativeCanvas.drawText("${box.model} ${box.label.uppercase()} ${(box.confidence * 100).toInt()}%", max(4f, cx - width / 2f), max(18f, cy - height / 2f - 4f), paint) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SettingsSheet(confidence: Float, onConfidence: (Float) -> Unit, analysisFps: Int, onAnalysisFps: (Float) -> Unit, maxObjects: Int, onMaxObjects: (Float) -> Unit, boxScale: Float, onBoxScale: (Float) -> Unit, onClose: () -> Unit, modelStatus: String) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = false)
    ModalBottomSheet(onDismissRequest = onClose, sheetState = sheetState, containerColor = Color(0xFF0B1018), dragHandle = { BottomSheetDefaults.DragHandle(color = Color.Cyan.copy(.75f)) }) {
        Column(Modifier.fillMaxWidth().navigationBarsPadding().padding(horizontal = 18.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Settings, null, tint = Color.Cyan)
                Spacer(Modifier.width(8.dp))
                Column(Modifier.weight(1f)) {
                    Text("AI CONTROL CENTER", color = Color.White, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text("Swipe panel • sliders apply instantly", color = Color.White.copy(.52f), style = MaterialTheme.typography.bodySmall)
                }
                IconButton(onClick = onClose) { Icon(Icons.Default.Close, "Close", tint = Color.White) }
            }
            Spacer(Modifier.height(8.dp))
            LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp), contentPadding = PaddingValues(bottom = 24.dp)) {
                item { SettingsCard("AI DETECTION") {
                    SmoothSlider("Confidence", confidence, .35f..0.90f, "${(confidence * 100).toInt()}%", onConfidence)
                    SmoothSlider("Analysis FPS", analysisFps.toFloat(), 5f..30f, "$analysisFps fps", onAnalysisFps)
                    SmoothSlider("Max objects", maxObjects.toFloat(), 1f..50f, "$maxObjects", onMaxObjects)
                } }
                item { SettingsCard("HUD") { SmoothSlider("Object box scale", boxScale, .30f..1.20f, "${(boxScale * 100).toInt()}%", onBoxScale) } }
                item { SettingsCard("MODEL") {
                    Text(modelStatus, color = Color.White.copy(.72f), style = MaterialTheme.typography.bodySmall)
                    Spacer(Modifier.height(6.dp))
                    Text("YOLO11x-Pose uses the imported .tflite model when available; otherwise ML Kit remains active.", color = Color.White.copy(.48f), style = MaterialTheme.typography.bodySmall)
                } }
            }
        }
    }
}

@Composable
private fun SettingsCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(shape = RoundedCornerShape(18.dp), colors = CardDefaults.cardColors(containerColor = Color.White.copy(.045f))) {
        Column(Modifier.padding(14.dp)) {
            Text(title, color = Color.Cyan.copy(.85f), fontWeight = FontWeight.Bold, style = MaterialTheme.typography.labelMedium)
            Spacer(Modifier.height(8.dp))
            content()
        }
    }
}

@Composable
private fun SmoothSlider(title: String, value: Float, range: ClosedFloatingPointRange<Float>, display: String, onChange: (Float) -> Unit) {
    val animatedValue by animateFloatAsState(value.coerceIn(range.start, range.endInclusive), spring(dampingRatio = .90f, stiffness = 650f), label = title)
    Row {
        Text(title, color = Color.White, modifier = Modifier.weight(1f))
        Text(display, color = Color.Cyan, fontWeight = FontWeight.Bold)
    }
    Slider(value = animatedValue, onValueChange = onChange, valueRange = range, modifier = Modifier.fillMaxWidth())
}
