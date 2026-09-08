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
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import org.json.JSONArray
import org.json.JSONObject
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

data class BoxData(
    val rect: RectF,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val rotationDegrees: Int,
    val label: String,
    val confidence: Float
)

enum class DeviceTier { LOW, MID, FLAGSHIP }

data class SavedDetection(
    val timestamp: Long,
    val label: String,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

private object DetectionStore {
    private const val PREFS = "ai_fusion_detection_history"
    private const val KEY = "detections"
    private const val MAX_RECORDS = 10000

    @Synchronized
    fun save(context: Context, detections: List<BoxData>) {
        if (detections.isEmpty()) return
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val old = try {
            JSONArray(prefs.getString(KEY, "[]") ?: "[]")
        } catch (_: Exception) {
            JSONArray()
        }
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
        for (i in start until old.length()) {
            result.put(old.getJSONObject(i))
        }
        prefs.edit().putString(KEY, result.toString()).apply()
    }

    fun count(context: Context): Int = try {
        JSONArray(
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getString(KEY, "[]") ?: "[]"
        ).length()
    } catch (_: Exception) {
        0
    }

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

private data class UpdateInfo(
    val available: Boolean,
    val version: String? = null,
    val releaseUrl: String? = null,
    val error: String? = null
)

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
    val connection = (URL(
        "https://api.github.com/repos/badru827i/-Ai-fusion-camera-/releases/latest"
    ).openConnection() as HttpURLConnection).apply {
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
        UpdateInfo(
            compareVersions(currentVersion(context), latest) < 0,
            latest,
            root.optString("html_url")
        )
    }
} catch (_: Exception) {
    UpdateInfo(false, error = "Unable to check for updates")
}

@Composable
private fun AiFusionCamera() {
    val context = LocalContext.current
    val profile = remember { tier(context) }
    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    var settings by remember { mutableStateOf(false) }
    var confidence by remember { mutableFloatStateOf(.55f) }
    var targetUiScale by remember { mutableFloatStateOf(.75f) }
    var controlsVisible by remember { mutableStateOf(true) }
    var boxes by remember { mutableStateOf(emptyList<BoxData>()) }
    var savedCount by remember { mutableIntStateOf(DetectionStore.count(context)) }
    var updateInfo by remember { mutableStateOf<UpdateInfo?>(null) }
    var checkingUpdate by remember { mutableStateOf(false) }

    val lastPersistAt = remember { AtomicLong(0L) }
    val storageExecutor = remember { Executors.newSingleThreadExecutor() }

    val cameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted = it }
    val files = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { }
    val tree = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { }

    DisposableEffect(Unit) {
        onDispose { storageExecutor.shutdown() }
    }

    fun persistDetections(detected: List<BoxData>) {
        if (detected.isEmpty()) return
        val now = System.currentTimeMillis()
        val previous = lastPersistAt.get()
        if (now - previous < 750L) return
        if (!lastPersistAt.compareAndSet(previous, now)) return
        savedCount = (savedCount + detected.size).coerceAtMost(10000)
        storageExecutor.execute {
            DetectionStore.save(context, detected)
        }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (granted) {
            CameraPreview(profile, confidence) { detected ->
                boxes = detected
                persistDetections(detected)
            }
            BoxOverlay(boxes, Modifier.fillMaxSize(), targetUiScale)
        } else {
            Column(
                Modifier.align(Alignment.Center),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Camera permission diperlukan", color = Color.White)
                Spacer(Modifier.height(12.dp))
                Button(onClick = { cameraPermission.launch(Manifest.permission.CAMERA) }) {
                    Text("Allow Camera")
                }
            }
        }

        Column(
            Modifier.fillMaxWidth().align(Alignment.TopCenter).padding(14.dp)
        ) {
            Text(
                "AI FUSION CAMERA",
                color = Color.White,
                style = MaterialTheme.typography.titleLarge
            )
            Text(
                "V3  •  ${profile.name}  •  AUTO ROTATION",
                color = Color.White.copy(.8f)
            )
        }

        FilledTonalButton(
            onClick = { controlsVisible = !controlsVisible },
            modifier = Modifier.align(Alignment.BottomEnd).padding(12.dp),
            contentPadding = PaddingValues(horizontal = 14.dp, vertical = 10.dp)
        ) {
            Text(if (controlsVisible) "Hide UI" else "Show UI")
        }

        if (controlsVisible) {
            Card(
                Modifier.fillMaxWidth().align(Alignment.BottomCenter)
                    .padding(start = 12.dp, end = 92.dp, bottom = 12.dp),
                shape = RoundedCornerShape(18.dp)
            ) {
                Row(
                    Modifier.fillMaxWidth().padding(8.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    Button(onClick = {
                        files.launch(arrayOf("image/*", "video/*", "application/*", "text/*"))
                    }) { Text("Files") }
                    Button(onClick = { tree.launch(null) }) { Text("USB / Drive") }
                    Button(onClick = { settings = true }) { Text("Settings") }
                }
            }
        }
    }

    if (settings) {
        Settings(
            confidence = confidence,
            onConfidence = { confidence = it },
            targetUiScale = targetUiScale,
            onTargetUiScale = { targetUiScale = it },
            controlsVisible = controlsVisible,
            onControlsVisible = { controlsVisible = it },
            savedCount = savedCount,
            onClearData = {
                DetectionStore.clear(context)
                savedCount = 0
            },
            currentVersion = currentVersion(context),
            updateInfo = updateInfo,
            checkingUpdate = checkingUpdate,
            onCheckUpdate = {
                checkingUpdate = true
                Thread {
                    val result = checkForUpdate(context)
                    Handler(context.mainLooper).post {
                        updateInfo = result
                        checkingUpdate = false
                    }
                }.start()
            },
            onOpenUpdate = { url ->
                context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
            },
            close = { settings = false }
        ) {
            context.startActivity(Intent(android.provider.Settings.ACTION_BLUETOOTH_SETTINGS))
        }
    }
}

@Composable
private fun CameraPreview(
    profile: DeviceTier,
    confidence: Float,
    onBoxes: (List<BoxData>) -> Unit
) {
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

    DisposableEffect(Unit) {
        onDispose {
            detector.close()
            executor.shutdown()
        }
    }

    AndroidView(
        modifier = Modifier.fillMaxSize(),
        factory = { ctx ->
            val view = PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
            val future = ProcessCameraProvider.getInstance(ctx)
            future.addListener({
                val provider = future.get()
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = view.surfaceProvider
                }

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
                    val minInterval = when (profile) {
                        DeviceTier.FLAGSHIP -> 55L
                        DeviceTier.MID -> 75L
                        DeviceTier.LOW -> 110L
                    }
                    if (now - previous < minInterval || !lastAnalysisAt.compareAndSet(previous, now)) {
                        proxy.close()
                        return@setAnalyzer
                    }

                    val image = proxy.image
                    if (image == null) {
                        proxy.close()
                        return@setAnalyzer
                    }

                    val rotation = proxy.imageInfo.rotationDegrees
                    detector.process(InputImage.fromMediaImage(image, rotation))
                        .addOnSuccessListener { result ->
                            val detected = result.mapNotNull { objectResult ->
                                val best = objectResult.labels.maxByOrNull { it.confidence }
                                val score = best?.confidence ?: 0f
                                if (score < confidence) return@mapNotNull null
                                BoxData(
                                    RectF(objectResult.boundingBox),
                                    image.width,
                                    image.height,
                                    rotation,
                                    best?.text ?: "OBJECT",
                                    score
                                )
                            }
                            onBoxes(detected)
                        }
                        .addOnCompleteListener { proxy.close() }
                }

                try {
                    provider.unbindAll()
                    provider.bindToLifecycle(
                        context as ComponentActivity,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis
                    )
                } catch (_: Exception) {
                    // Camera can disappear during activity recreation; keep UI alive.
                }
            }, ContextCompat.getMainExecutor(ctx))
            view
        }
    )
}

@Composable
private fun BoxOverlay(
    boxes: List<BoxData>,
    modifier: Modifier,
    uiScale: Float
) {
    Canvas(modifier) {
        boxes.forEach { box ->
            val sourceW = if (box.rotationDegrees % 180 == 0) {
                box.sourceWidth
            } else {
                box.sourceHeight
            }
            val sourceH = if (box.rotationDegrees % 180 == 0) {
                box.sourceHeight
            } else {
                box.sourceWidth
            }
            val scale = max(
                size.width / sourceW.toFloat(),
                size.height / sourceH.toFloat()
            )
            val cropX = (sourceW * scale - size.width) / 2f
            val cropY = (sourceH * scale - size.height) / 2f
            val r = box.rect
            val centerX = r.centerX() * scale - cropX
            val centerY = r.centerY() * scale - cropY
            val width = (r.width() * scale * uiScale).coerceAtLeast(2f)
            val height = (r.height() * scale * uiScale).coerceAtLeast(2f)
            val stroke = (
                1.5.dp.toPx() * uiScale.coerceIn(.45f, 1.2f)
            ).coerceAtLeast(1f)

            drawRect(
                Color.Cyan,
                Offset(centerX - width / 2f, centerY - height / 2f),
                androidx.compose.ui.geometry.Size(width, height),
                style = Stroke(stroke)
            )

            val labelText = if (box.confidence > 0f) {
                "${box.label.uppercase()} ${(box.confidence * 100).toInt()}%"
            } else {
                box.label.uppercase()
            }
            val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
                color = android.graphics.Color.WHITE
                textSize = (
                    12.dp.toPx() * uiScale.coerceIn(.55f, 1.0f)
                ).coerceAtLeast(8.dp.toPx())
                typeface = android.graphics.Typeface.create(
                    android.graphics.Typeface.DEFAULT,
                    android.graphics.Typeface.BOLD
                )
            }
            val fm = paint.fontMetrics
            val textX = (centerX - width / 2f).coerceAtLeast(4f)
            val textY = (
                centerY - height / 2f - 5.dp.toPx()
            ).coerceAtLeast(-fm.top + 2f)

            drawIntoCanvas { canvas ->
                canvas.nativeCanvas.drawText(labelText, textX, textY, paint)
            }
        }
    }
}

@Composable
private fun Settings(
    confidence: Float,
    onConfidence: (Float) -> Unit,
    targetUiScale: Float,
    onTargetUiScale: (Float) -> Unit,
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
    AlertDialog(
        onDismissRequest = close,
        title = { Text("Ai Fusion Camera Settings") },
        text = {
            Column {
                Text("AI confidence ${(confidence * 100).toInt()}%")
                Slider(
                    value = confidence,
                    onValueChange = onConfidence,
                    valueRange = .35f..0.9f
                )
                Spacer(Modifier.height(8.dp))
                Text("Target Object UI: ${(targetUiScale * 100).toInt()}%")
                Text(
                    "Kotak + text detection dikecilkan. Default 75%.",
                    style = MaterialTheme.typography.bodySmall
                )
                Slider(
                    value = targetUiScale,
                    onValueChange = onTargetUiScale,
                    valueRange = .45f..1.2f
                )
                Spacer(Modifier.height(8.dp))
                Text("Detection data disimpan: $savedCount rekod")
                Text(
                    "Label, confidence, masa dan koordinat petak disimpan dalam telefon; tiada server diperlukan.",
                    style = MaterialTheme.typography.bodySmall
                )
                Spacer(Modifier.height(4.dp))
                OutlinedButton(
                    onClick = onClearData,
                    modifier = Modifier.fillMaxWidth(),
                    enabled = savedCount > 0
                ) {
                    Text("Clear Saved Detection Data")
                }
                Spacer(Modifier.height(8.dp))
                Text("External buttons")
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Switch(
                        checked = controlsVisible,
                        onCheckedChange = onControlsVisible
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        if (controlsVisible) {
                            "Files / USB / Settings: ON"
                        } else {
                            "Files / USB / Settings: OFF"
                        }
                    )
                }
                Spacer(Modifier.height(8.dp))
                Text("App version: $currentVersion")
                Button(
                    onClick = onCheckUpdate,
                    enabled = !checkingUpdate,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(if (checkingUpdate) "Checking..." else "Check for Update")
                }
                updateInfo?.let { info ->
                    if (info.available && info.version != null && info.releaseUrl != null) {
                        Text(
                            "Update available: v${info.version}",
                            color = MaterialTheme.colorScheme.primary
                        )
                        Button(
                            onClick = { onOpenUpdate(info.releaseUrl) },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Open Update")
                        }
                    } else if (info.error != null) {
                        Text(info.error, style = MaterialTheme.typography.bodySmall)
                    } else {
                        Text(
                            "You are using the latest version.",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
                Spacer(Modifier.height(8.dp))
                Text("Orientation: Auto • 16:9 / 9:16")
                Text("Performance: automatic device profile + frame throttling")
                Text("Files: Android Storage Access Framework")
                Text("USB OTG / pendrive: supported when Android exposes the drive")
                Spacer(Modifier.height(10.dp))
                Button(onClick = bluetooth, modifier = Modifier.fillMaxWidth()) {
                    Text("Connect Bluetooth")
                }
            }
        },
        confirmButton = { TextButton(onClick = close) { Text("Done") } }
    )
}
