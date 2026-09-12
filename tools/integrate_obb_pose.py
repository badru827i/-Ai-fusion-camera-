from pathlib import Path

p = Path('app/src/main/java/com/badru/aifusioncamera/MainActivity.kt')
s = p.read_text()
if 'YoloObbDetector.load(context)' in s:
    print('OBB + Pose integration already present')
    raise SystemExit(0)

s = s.replace('import kotlin.math.max\n', 'import kotlin.math.max\nimport kotlin.math.PI\nimport androidx.compose.ui.graphics.drawscope.rotate\n')
s = s.replace('data class BoxData(val rect: RectF, val sourceWidth: Int, val sourceHeight: Int, val rotationDegrees: Int, val label: String, val confidence: Float)', '''data class BoxData(
    val rect: RectF, val sourceWidth: Int, val sourceHeight: Int, val rotationDegrees: Int,
    val label: String, val confidence: Float, val angle: Float = 0f,
    val keypoints: FloatArray? = null, val model: String = "AI"
)''')
old = '''    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        val fileName = context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
        if (fileName?.lowercase()?.endsWith(".tflite") != true) {
            modelText = "Import gagal • pilih fail .tflite"
            return@rememberLauncherForActivityResult
        }
        try {
            val dir = File(context.filesDir, "models").apply { mkdirs() }
            val target = File(dir, YoloPoseDetector.MODEL_NAME)
            context.contentResolver.openInputStream(uri)?.use { input -> target.outputStream().use { out -> input.copyTo(out) } }
            modelGeneration++
            modelText = "YOLO11x-Pose imported • ${target.length() / (1024 * 1024)} MB"
        } catch (_: Exception) {
            modelText = "Import gagal"
        }
    }
'''
new = '''    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
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
'''
if old not in s: raise SystemExit('import block not found')
s = s.replace(old, new)
old_call = '''                CameraPreview(profile, confidence, analysisFps, maxObjects,
                    onStatus = { active ->
                        yoloReady = active
                        modelText = if (active) "YOLO11x-Pose • ACTIVE • 17 keypoints" else "ML Kit fallback • YOLO model not loaded"
                    },
                    onBoxes = { boxes = it })'''
new_call = '''                CameraPreview(profile, confidence, analysisFps, maxObjects,
                    onStatus = { poseActive, obbActive ->
                        yoloReady = poseActive || obbActive
                        modelText = when {
                            poseActive && obbActive -> "YOLO Pose + OBB • ACTIVE"
                            poseActive -> "YOLO11x-Pose • ACTIVE • 17 keypoints"
                            obbActive -> "YOLO-OBB • ACTIVE • rotated boxes"
                            else -> "ML Kit fallback • YOLO models not loaded"
                        }
                    },
                    onBoxes = { boxes = it })'''
if old_call not in s: raise SystemExit('camera call not found')
s = s.replace(old_call, new_call)
start = s.index('@Composable\nprivate fun CameraPreview')
end = s.index('@Composable\nprivate fun BoxOverlay', start)
new_camera = '''@Composable
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

'''
s = s[:start] + new_camera + s[end:]
start = s.index('@Composable\nprivate fun BoxOverlay')
end = s.index('@OptIn(ExperimentalMaterial3Api::class)', start)
new_overlay = '''@Composable
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

'''
s = s[:start] + new_overlay + s[end:]
p.write_text(s)

obb_path = Path('app/src/main/java/com/badru/aifusioncamera/YoloObbDetector.kt')
o = obb_path.read_text()
o = o.replace('val angle = if (attrCount > 5 + classCount) value(i, 4 + classCount) else 0f', 'val angle = value(i, attrCount - 1)')
obb_path.write_text(o)

pose_path = Path('app/src/main/java/com/badru/aifusioncamera/YoloPoseDetector.kt')
o = pose_path.read_text()
old_load_start = o.index('        fun load(context: Context): YoloPoseDetector? = try {')
old_load_end = o.index('        private fun fromFile(file: File): YoloPoseDetector {', old_load_start)
new_load = '''        fun load(context: Context): YoloPoseDetector? = try {
            val candidates = File(context.filesDir, "models").listFiles()
                ?.filter { it.extension.equals("tflite", ignoreCase = true) }
                ?.filter { ModelManager.kindForFileName(it.name) == ModelManager.ModelKind.POSE }
                ?.sortedBy { it.name }
                .orEmpty()
            val active = candidates.firstOrNull { ModelManager.register(context, it).state == ModelManager.ModelState.ACTIVE }
            active?.let { fromFile(it) }
        } catch (_: Throwable) { null }

'''
o = o[:old_load_start] + new_load + o[old_load_end:]
pose_path.write_text(o)
print('Integration applied')
