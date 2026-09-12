package com.badru.aifusioncamera

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.os.Handler
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
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { AiFusionCamera() }
    }
}

data class BoxData(val rect: RectF, val sourceWidth: Int, val sourceHeight: Int, val rotationDegrees: Int, val label: String, val confidence: Float)
enum class DeviceTier { LOW, MID, FLAGSHIP }

data class DetectionStat(val label: String, val count: Int, val confidence: Float)

data class AiUltraConfig(
    val confidence: Float = .55f,
    val targetUiScale: Float = .60f,
    val analysisFps: Int = 15,
    val maxObjects: Int = 10,
    val saveIntervalMs: Long = 750L
)

private object DetectionStore {
    private const val PREFS = "ai_fusion_detection_history"
    private const val KEY = "detections"
    private const val MAX_RECORDS = 10000

    @Synchronized
    fun save(context: Context, detections: List<BoxData>) {
        if (detections.isEmpty()) return
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val old = try { JSONArray(prefs.getString(KEY, "[]") ?: "[]") } catch (_: Exception) { JSONArray() }
        val now = System.currentTimeMillis()
        detections.forEach { d ->
            old.put(JSONObject().apply {
                put("timestamp", now)
                put("label", d.label)
                put("confidence", d.confidence)
                put("left", d.rect.left)
                put("top", d.rect.top)
                put("right", d.rect.right)
                put("bottom", d.rect.bottom)
            })
        }
        val start = (old.length() - MAX_RECORDS).coerceAtLeast(0)
        val result = JSONArray()
        for (i in start until old.length()) result.put(old.getJSONObject(i))
        prefs.edit().putString(KEY, result.toString()).apply()
    }

    fun count(context: Context): Int = try {
        JSONArray(context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, "[]") ?: "[]").length()
    } catch (_: Exception) { 0 }

    fun clear(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove(KEY).apply()
    }
}

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

private data class UpdateInfo(val available: Boolean, val version: String? = null, val releaseUrl: String? = null, val error: String? = null)

private fun currentVersion(context: Context): String =
    context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "3.0.0"

private fun compareVersions(current: String, latest: String): Int {
    val a = current.removePrefix("v").split(".").map { it.toIntOrNull() ?: 0 }
    val b = latest.removePrefix("v").split(".").map { it.toIntOrNull() ?: 0 }
    for (i in 0 until max(a.size, b.size)) {
        val av = a.getOrElse(i) { 0 }
        val bv = b.getOrElse(i) { 0 }
        if (av != bv) return av.compareTo(bv)
    }
    return 0
}

private fun checkForUpdate(context: Context): UpdateInfo = try {
    val connection = (URL("https://api.github.com/repos/badru827i/-Ai-fusion-camera-/releases/latest").openConnection() as HttpURLConnection).apply {
        requestMethod = "GET"
        connectTimeout = 8000
        readTimeout = 8000
        setRequestProperty("Accept", "application/vnd.github+json")
        setRequestProperty("User-Agent", "Ai-Fusion-Camera")
    }
    if (connection.responseCode !in 200..299) {
        connection.disconnect()
        UpdateInfo(false, error = "No published release yet")
    } else {
        val root = connection.inputStream.bufferedReader().use { JSONObject(it.readText()) }
        connection.disconnect()
        val latest = root.optString("tag_name").removePrefix("v")
        UpdateInfo(compareVersions(currentVersion(context), latest) < 0, latest, root.optString("html_url"))
    }
} catch (_: Exception) {
    UpdateInfo(false, error = "Unable to check for updates")
}

@Composable
private fun AiFusionCamera() {
    val context = LocalContext.current
    val profile = remember { tier(context) }
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    var settings by remember { mutableStateOf(false) }
    var analyticsVisible by remember { mutableStateOf(true) }
    var confidence by remember { mutableFloatStateOf(.55f) }
    var targetUiScale by remember { mutableFloatStateOf(.60f) }
    var analysisFps by remember { mutableIntStateOf(15) }
    var maxObjects by remember { mutableIntStateOf(10) }
    var saveIntervalMs by remember { mutableLongStateOf(750L) }
    var controlsVisible by remember { mutableStateOf(true) }
    var boxes by remember { mutableStateOf(emptyList<BoxData>()) }
    var graphHistory by remember { mutableStateOf(listOf(0)) }
    var savedCount by remember { mutableIntStateOf(DetectionStore.count(context)) }
    var updateInfo by remember { mutableStateOf<UpdateInfo?>(null) }
    var checkingUpdate by remember { mutableStateOf(false) }
    var modelGeneration by remember { mutableIntStateOf(0) }
    var modelStatus by remember { mutableStateOf("YOLO11x-Pose: model belum dimuat • ML Kit fallback") }
    val lastPersistAt = remember { AtomicLong(0L) }
    val storageExecutor = remember { Executors.newSingleThreadExecutor() }
    val cameraPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    val files = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        val name = context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
        if (name?.lowercase()?.endsWith(".tflite") != true) {
            modelStatus = "Pilih fail YOLO .tflite"
            return@rememberLauncherForActivityResult
        }
        try {
            val dir = File(context.filesDir, "models").apply { mkdirs() }
            val target = File(dir, YoloPoseDetector.MODEL_NAME)
            context.contentResolver.openInputStream(uri)?.use { input -> target.outputStream().use { output -> input.copyTo(output) } }
            modelGeneration++
            modelStatus = "YOLO11x-Pose dimuat • ${target.length() / (1024 * 1024)} MB"
        } catch (_: Exception) {
            modelStatus = "Gagal import model YOLO"
        }
    }
    val tree = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { }

    DisposableEffect(Unit) { onDispose { storageExecutor.shutdown() } }

    fun persistDetections(detected: List<BoxData>) {
        if (detected.isEmpty()) return
        val now = System.currentTimeMillis()
        val previous = lastPersistAt.get()
        if (now - previous < saveIntervalMs || !lastPersistAt.compareAndSet(previous, now)) return
        savedCount = (savedCount + detected.size).coerceAtMost(10000)
        storageExecutor.execute { DetectionStore.save(context, detected) }
    }

    LaunchedEffect(boxes.size) {
        graphHistory = (graphHistory + boxes.size).takeLast(28)
    }

    val stats = remember(boxes) {
        boxes.groupBy { it.label.uppercase() }
            .map { (label, list) -> DetectionStat(label, list.size, list.map { it.confidence }.average().toFloat()) }
            .sortedByDescending { it.count }
    }
    val totalObjects = boxes.size
    val avgConfidence = if (boxes.isEmpty()) 0f else boxes.map { it.confidence }.average().toFloat()

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (granted) {
            CameraPreview(profile, confidence, analysisFps, maxObjects, modelGeneration) { detected, yoloActive ->
                boxes = detected
                modelStatus = if (yoloActive) "YOLO11x-Pose • ACTIVE • 17 keypoints" else if (modelStatus.startsWith("YOLO11x-Pose: ACTIVE")) modelStatus else "ML Kit realtime fallback"
                persistDetections(detected)
            }
            BoxOverlay(boxes, Modifier.fillMaxSize(), targetUiScale)
        } else {
            Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
                Text("Camera permission diperlukan", color = Color.White)
                Spacer(Modifier.height(12.dp))
                Button(onClick = { cameraPermission.launch(Manifest.permission.CAMERA) }) { Text("Allow Camera") }
            }
        }

        // Premium camera header.
        Surface(
            modifier = Modifier.align(Alignment.TopCenter).padding(12.dp),
            color = Color.Black.copy(alpha = .58f),
            shape = RoundedCornerShape(20.dp),
            tonalElevation = 6.dp
        ) {
            Row(Modifier.padding(horizontal = 14.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Visibility, null, tint = Color.Cyan, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Column {
                    Text("NEXA VISION", color = Color.White, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Text("AI CAMERA  •  ${profile.name}", color = Color.White.copy(.68f), style = MaterialTheme.typography.labelSmall)
                }
                Spacer(Modifier.width(12.dp))
                Surface(color = Color(0x3322FFAA), shape = RoundedCornerShape(999.dp)) {
                    Text("● LIVE", color = Color(0xFF55FFB0), style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
                }
            }
        }

        // Live analytics panel: total + graph + every detected object.
        AnimatedVisibility(
            visible = analyticsVisible,
            enter = fadeIn(spring()) ,
            exit = fadeOut(),
            modifier = Modifier.align(Alignment.TopEnd).padding(top = 82.dp, end = 10.dp)
        ) {
            AnalyticsPanel(
                totalObjects = totalObjects,
                avgConfidence = avgConfidence,
                stats = stats,
                graphHistory = graphHistory
            )
        }

        // Bottom action dock.
        Surface(
            modifier = Modifier.align(Alignment.BottomCenter).padding(horizontal = 10.dp, vertical = 10.dp),
            color = Color.Black.copy(alpha = .70f),
            shape = RoundedCornerShape(22.dp),
            tonalElevation = 8.dp
        ) {
            Column(Modifier.padding(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Surface(color = Color(0x241AA6FF), shape = RoundedCornerShape(999.dp)) {
                        Text("DETECTED  $totalObjects", color = Color.Cyan, style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp))
                    }
                    Spacer(Modifier.weight(1f))
                    IconButton(onClick = { analyticsVisible = !analyticsVisible }) { Icon(Icons.Default.Analytics, "Toggle analytics", tint = Color.White) }
                    IconButton(onClick = { settings = true }) { Icon(Icons.Default.Tune, "Open settings", tint = Color.White) }
                }
                AnimatedVisibility(visible = controlsVisible) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        Button(onClick = { files.launch(arrayOf("application/octet-stream", "*/*")) }, shape = RoundedCornerShape(14.dp)) { Text("Import YOLO") }
                        Button(onClick = { tree.launch(null) }, shape = RoundedCornerShape(14.dp)) { Text("Drive") }
                        Button(onClick = { controlsVisible = false }, shape = RoundedCornerShape(14.dp)) { Text("Minimal") }
                    }
                }
                if (!controlsVisible) {
                    TextButton(onClick = { controlsVisible = true }, modifier = Modifier.align(Alignment.CenterHorizontally)) { Text("Show controls") }
                }
            }
        }
    }

    if (settings) {
        SettingsSheet(
            confidence = confidence,
            onConfidence = { confidence = it },
            targetUiScale = targetUiScale,
            onTargetUiScale = { targetUiScale = it },
            analysisFps = analysisFps,
            onAnalysisFps = { analysisFps = it },
            maxObjects = maxObjects,
            onMaxObjects = { maxObjects = it },
            saveIntervalMs = saveIntervalMs,
            onSaveInterval = { saveIntervalMs = it },
            controlsVisible = controlsVisible,
            onControlsVisible = { controlsVisible = it },
            savedCount = savedCount,
            onClearData = { DetectionStore.clear(context); savedCount = 0 },
            currentVersion = currentVersion(context),
            updateInfo = updateInfo,
            checkingUpdate = checkingUpdate,
            onCheckUpdate = {
                checkingUpdate = true
                Thread {
                    val result = checkForUpdate(context)
                    Handler(context.mainLooper).post { updateInfo = result; checkingUpdate = false }
                }.start()
            },
            onOpenUpdate = { url -> context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) },
            close = { settings = false },
            bluetooth = { context.startActivity(Intent(android.provider.Settings.ACTION_BLUETOOTH_SETTINGS)) }
        )
    }
}

@Composable
private fun AnalyticsPanel(
    totalObjects: Int,
    avgConfidence: Float,
    stats: List<DetectionStat>,
    graphHistory: List<Int>
) {
    Card(
        modifier = Modifier.widthIn(min = 240.dp, max = 300.dp),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = Color.Black.copy(alpha = .70f)),
        border = CardDefaults.outlinedCardBorder().copy(alpha = .45f)
    ) {
        Column(Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Analytics, null, tint = Color.Cyan, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(7.dp))
                Text("AI DETECTION", color = Color.White, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricChip("TOTAL", totalObjects.toString())
                MetricChip("CONF", "${(avgConfidence * 100).toInt()}%")
            }
            Spacer(Modifier.height(10.dp))
            Text("REALTIME GRAPH", color = Color.White.copy(.62f), style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.height(4.dp))
            DetectionGraph(values = graphHistory, modifier = Modifier.fillMaxWidth().height(64.dp))
            Spacer(Modifier.height(9.dp))
            Text("OBJECTS", color = Color.White.copy(.62f), style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.height(4.dp))
            if (stats.isEmpty()) {
                Text("No object detected", color = Color.White.copy(.55f), style = MaterialTheme.typography.bodySmall)
            } else {
                LazyColumn(modifier = Modifier.heightIn(max = 150.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    items(stats.take(8), key = { it.label }) { stat ->
                        ObjectStatRow(stat)
                    }
                }
            }
        }
    }
}

@Composable
private fun MetricChip(title: String, value: String) {
    Surface(color = Color.White.copy(alpha = .08f), shape = RoundedCornerShape(12.dp)) {
        Row(Modifier.padding(horizontal = 9.dp, vertical = 7.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(title, color = Color.White.copy(.55f), style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.width(5.dp))
            Text(value, color = Color.Cyan, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun ObjectStatRow(stat: DetectionStat) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Surface(color = Color.Cyan.copy(alpha = .12f), shape = RoundedCornerShape(9.dp)) {
            Text(stat.label, color = Color.White, style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp))
        }
        Spacer(Modifier.width(6.dp))
        Text("×${stat.count}", color = Color.Cyan, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
        Spacer(Modifier.weight(1f))
        Text("${(stat.confidence * 100).toInt()}%", color = Color.White.copy(.72f), style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun DetectionGraph(values: List<Int>, modifier: Modifier = Modifier) {
    val maxValue = (values.maxOrNull() ?: 1).coerceAtLeast(1)
    Canvas(modifier) {
        if (values.size < 2) return@Canvas
        val stepX = size.width / (values.size - 1).toFloat()
        val points = values.mapIndexed { index, value ->
            Offset(
                x = index * stepX,
                y = size.height - (value / maxValue.toFloat()) * size.height
            )
        }
        for (i in 1 until points.size) {
            drawLine(Color.Cyan.copy(alpha = .75f), points[i - 1], points[i], strokeWidth = 3f)
        }
        points.forEach { p -> drawCircle(Color.Cyan, radius = 3.2f, center = p) }
    }
}

@Composable
private fun CameraPreview(
    profile: DeviceTier,
    confidence: Float,
    analysisFps: Int,
    maxObjects: Int,
    modelGeneration: Int,
    onBoxes: (List<BoxData>, Boolean) -> Unit
) {
    val context = LocalContext.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    val yolo = remember(modelGeneration) { YoloPoseDetector.load(context) }
    val detector = remember {
        ObjectDetection.getClient(
            ObjectDetectorOptions.Builder()
                .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
                .enableMultipleObjects()
                .enableClassification()
                .build()
        )
    }
    DisposableEffect(yolo) { onDispose { yolo?.close(); detector.close(); executor.shutdown() } }
    AndroidView(modifier = Modifier.fillMaxSize(), factory = { ctx ->
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
            val lastAnalysisAt = AtomicLong(0L)
            analysis.setAnalyzer(executor) { proxy ->
                val now = System.currentTimeMillis()
                val previous = lastAnalysisAt.get()
                val minInterval = (1000L / analysisFps.coerceIn(5, 30)).coerceAtLeast(33L)
                if (now - previous < minInterval || !lastAnalysisAt.compareAndSet(previous, now)) {
                    proxy.close()
                    return@setAnalyzer
                }
                if (yolo != null) {
                    try {
                        val poses = yolo.detect(proxy, confidence)
                        val sourceW = if (proxy.imageInfo.rotationDegrees % 180 == 0) proxy.width else proxy.height
                        val sourceH = if (proxy.imageInfo.rotationDegrees % 180 == 0) proxy.height else proxy.width
                        val detected = poses.take(maxObjects.coerceIn(1, 20)).map { pose ->
                            BoxData(
                                RectF(
                                    pose.rect.left * sourceW,
                                    pose.rect.top * sourceH,
                                    pose.rect.right * sourceW,
                                    pose.rect.bottom * sourceH
                                ),
                                sourceW,
                                sourceH,
                                0,
                                "PERSON",
                                pose.confidence
                            )
                        }
                        onBoxes(detected, true)
                    } catch (_: Throwable) {
                        onBoxes(emptyList(), true)
                    } finally {
                        proxy.close()
                    }
                    return@setAnalyzer
                }
                val image = proxy.image ?: run { proxy.close(); return@setAnalyzer }
                val rotation = proxy.imageInfo.rotationDegrees
                detector.process(InputImage.fromMediaImage(image, rotation)).addOnSuccessListener { result ->
                    val detected = result.mapNotNull { objectResult ->
                        val best = objectResult.labels.maxByOrNull { it.confidence }
                        val score = best?.confidence ?: 0f
                        if (score < confidence) null
                        else BoxData(RectF(objectResult.boundingBox), image.width, image.height, rotation, best?.text ?: "OBJECT", score)
                    }.sortedByDescending { it.confidence }.take(maxObjects.coerceIn(1, 50))
                    onBoxes(detected, false)
                }.addOnCompleteListener { proxy.close() }
            }
            try {
                provider.unbindAll()
                provider.bindToLifecycle(context as ComponentActivity, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (_: Exception) { }
        }, ContextCompat.getMainExecutor(ctx))
        view
    })
}

@Composable
private fun BoxOverlay(boxes: List<BoxData>, modifier: Modifier, uiScale: Float) {
    Canvas(modifier) {
        boxes.forEach { box ->
            val sourceW = if (box.rotationDegrees % 180 == 0) box.sourceWidth else box.sourceHeight
            val sourceH = if (box.rotationDegrees % 180 == 0) box.sourceHeight else box.sourceWidth
            val scale = max(size.width / sourceW.toFloat(), size.height / sourceH.toFloat())
            val cropX = (sourceW * scale - size.width) / 2f
            val cropY = (sourceH * scale - size.height) / 2f
            val r = box.rect
            val centerX = r.centerX() * scale - cropX
            val centerY = r.centerY() * scale - cropY
            val targetScale by animateFloatAsState(
                targetValue = uiScale.coerceIn(.30f, 1.2f),
                animationSpec = spring(dampingRatio = .82f, stiffness = 420f),
                label = "box-scale"
            )
            val width = (r.width() * scale * targetScale).coerceAtLeast(2f)
            val height = (r.height() * scale * targetScale).coerceAtLeast(2f)
            val stroke = (1.5.dp.toPx() * targetScale.coerceIn(.35f, 1.2f)).coerceAtLeast(1f)
            drawRect(
                Color.Cyan.copy(alpha = .92f),
                Offset(centerX - width / 2f, centerY - height / 2f),
                androidx.compose.ui.geometry.Size(width, height),
                style = Stroke(stroke)
            )
            val labelText = "${box.label.uppercase()} ${(box.confidence * 100).toInt()}%"
            val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
                color = android.graphics.Color.WHITE
                textSize = (12.dp.toPx() * targetScale.coerceIn(.45f, 1.0f)).coerceAtLeast(7.dp.toPx())
                typeface = android.graphics.Typeface.create(android.graphics.Typeface.DEFAULT, android.graphics.Typeface.BOLD)
            }
            val fm = paint.fontMetrics
            val textX = (centerX - width / 2f).coerceAtLeast(4f)
            val textY = (centerY - height / 2f - 4.dp.toPx()).coerceAtLeast(-fm.top + 2f)
            drawIntoCanvas { canvas -> canvas.nativeCanvas.drawText(labelText, textX, textY, paint) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SettingsSheet(
    confidence: Float,
    onConfidence: (Float) -> Unit,
    targetUiScale: Float,
    onTargetUiScale: (Float) -> Unit,
    analysisFps: Int,
    onAnalysisFps: (Int) -> Unit,
    maxObjects: Int,
    onMaxObjects: (Int) -> Unit,
    saveIntervalMs: Long,
    onSaveInterval: (Long) -> Unit,
    controlsVisible: Boolean,
    onControlsVisible: (Boolean) -> Unit,
    savedCount: Int,
    onClearData: () -> Unit,
    currentVersion: String,
    updateInfo: UpdateInfo?,
    checkingUpdate: Boolean,
    onCheckUpdate: () -> Unit,
    onOpenUpdate: (String) -> Unit,
    close: () -> Unit,
    bluetooth: () -> Unit
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = false)
    ModalBottomSheet(
        onDismissRequest = close,
        sheetState = sheetState,
        containerColor = Color(0xFF0B0F16),
        dragHandle = { BottomSheetDefaults.DragHandle(color = Color.Cyan.copy(alpha = .75f)) }
    ) {
        Column(Modifier.fillMaxWidth().padding(horizontal = 18.dp).navigationBarsPadding()) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Settings, null, tint = Color.Cyan)
                Spacer(Modifier.width(8.dp))
                Column(Modifier.weight(1f)) {
                    Text("AI CONTROL CENTER", color = Color.White, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text("Swipe up/down • changes apply instantly", color = Color.White.copy(.58f), style = MaterialTheme.typography.bodySmall)
                }
                IconButton(onClick = close) { Icon(Icons.Default.Close, "Close settings", tint = Color.White) }
            }
            Spacer(Modifier.height(8.dp))
            LazyColumn(verticalArrangement = Arrangement.spacedBy(14.dp), contentPadding = PaddingValues(bottom = 22.dp)) {
                item {
                    SettingSection("AI DETECTION") {
                        SmoothSlider("Confidence", confidence, .35f..0.9f, "${(confidence * 100).toInt()}%") { onConfidence(it) }
                        SmoothSlider("Analysis FPS", analysisFps.toFloat(), 5f..30f, "$analysisFps fps") { onAnalysisFps(it.toInt().coerceIn(5, 30)) }
                        SmoothSlider("Max objects", maxObjects.toFloat(), 1f..50f, "$maxObjects") { onMaxObjects(it.toInt().coerceIn(1, 50)) }
                    }
                }
                item {
                    SettingSection("HUD & GRAPHICS") {
                        SmoothSlider("Object UI scale", targetUiScale, .30f..1.2f, "${(targetUiScale * 100).toInt()}%") { onTargetUiScale(it) }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Bottom controls", color = Color.White, modifier = Modifier.weight(1f))
                            Switch(checked = controlsVisible, onCheckedChange = onControlsVisible)
                        }
                    }
                }
                item {
                    SettingSection("STORAGE") {
                        SmoothSlider("Save interval", saveIntervalMs.toFloat(), 250f..5000f, "${saveIntervalMs} ms") { onSaveInterval(it.toLong().coerceIn(250L, 5000L)) }
                        Text("Saved detection records: $savedCount", color = Color.White.copy(.70f), style = MaterialTheme.typography.bodySmall)
                        OutlinedButton(onClick = onClearData, enabled = savedCount > 0, modifier = Modifier.fillMaxWidth()) { Text("Clear detection history") }
                    }
                }
                item {
                    SettingSection("SYSTEM") {
                        Text("Version $currentVersion", color = Color.White.copy(.70f))
                        Spacer(Modifier.height(6.dp))
                        Button(onClick = onCheckUpdate, enabled = !checkingUpdate, modifier = Modifier.fillMaxWidth()) {
                            Text(if (checkingUpdate) "Checking update..." else "Check for update")
                        }
                        updateInfo?.let { info ->
                            Spacer(Modifier.height(5.dp))
                            Text(if (info.available) "Update available: v${info.version}" else (info.error ?: "You are up to date"), color = Color.White.copy(.72f), style = MaterialTheme.typography.bodySmall)
                            if (info.available && info.releaseUrl != null) {
                                OutlinedButton(onClick = { onOpenUpdate(info.releaseUrl) }, modifier = Modifier.fillMaxWidth()) { Text("Open release") }
                            }
                        }
                        OutlinedButton(onClick = bluetooth, modifier = Modifier.fillMaxWidth()) { Text("Bluetooth settings") }
                        Spacer(Modifier.height(5.dp))
                        Text("Camera, network and model runtime stay independent of these UI controls.", color = Color.White.copy(.50f), style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}

@Composable
private fun SettingSection(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White.copy(alpha = .045f)),
        border = CardDefaults.outlinedCardBorder().copy(alpha = .20f)
    ) {
        Column(Modifier.padding(14.dp), content = {
            Text(title, color = Color.Cyan.copy(.85f), style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))
            content()
        })
    }
}

@Composable
private fun SmoothSlider(title: String, value: Float, range: ClosedFloatingPointRange<Float>, valueText: String, onValueChange: (Float) -> Unit) {
    val animatedValue by animateFloatAsState(
        targetValue = value,
        animationSpec = spring(dampingRatio = .9f, stiffness = 650f),
        label = title
    )
    Column {
        Row {
            Text(title, color = Color.White, modifier = Modifier.weight(1f))
            Text(valueText, color = Color.Cyan, fontWeight = FontWeight.Bold)
        }
        Slider(
            value = animatedValue,
            onValueChange = onValueChange,
            valueRange = range,
            modifier = Modifier.fillMaxWidth()
        )
    }
}
