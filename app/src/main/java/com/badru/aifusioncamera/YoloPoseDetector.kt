package com.badru.aifusioncamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class YoloPoseDetector private constructor(private val interpreter: Interpreter) : AutoCloseable {
    companion object {
        const val MODEL_NAME = "yolo11x-pose.tflite"
        private const val INPUT = 640
        private const val ATTR = 56
        private const val MAX = 8400

        fun load(context: Context): YoloPoseDetector? = try {
            val imported = File(context.filesDir, "models/$MODEL_NAME")
            val file = if (imported.exists() && imported.length() > 0) imported else null
            if (file != null) fromFile(file) else {
                val dir = File(context.filesDir, "models").apply { mkdirs() }
                val target = File(dir, MODEL_NAME)
                context.assets.open(MODEL_NAME).use { input -> target.outputStream().use { input.copyTo(it) } }
                fromFile(target)
            }
        } catch (_: Throwable) { null }

        private fun fromFile(file: File): YoloPoseDetector {
            val stream = FileInputStream(file)
            val mapped = stream.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
            stream.close()
            val options = Interpreter.Options().apply { setNumThreads(max(2, min(4, Runtime.getRuntime().availableProcessors()))) }
            return YoloPoseDetector(Interpreter(mapped, options))
        }
    }

    data class Pose(val rect: android.graphics.RectF, val confidence: Float, val keypoints: FloatArray)

    private val inputType = interpreter.getInputTensor(0).dataType()
    private val outputShape = interpreter.getOutputTensor(0).shape()

    fun detect(proxy: ImageProxy, threshold: Float): List<Pose> {
        val source = proxy.toBitmap() ?: return emptyList()
        val bitmap = Bitmap.createBitmap(INPUT, INPUT, Bitmap.Config.ARGB_8888)
        android.graphics.Canvas(bitmap).drawBitmap(source, null, android.graphics.Rect(0, 0, INPUT, INPUT), android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))
        source.recycle()
        val bytes = if (inputType == DataType.FLOAT32) 4 else 1
        val input = ByteBuffer.allocateDirect(INPUT * INPUT * 3 * bytes).order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT * INPUT)
        bitmap.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT)
        for (p in pixels) {
            val r = ((p shr 16) and 255) / 255f
            val g = ((p shr 8) and 255) / 255f
            val b = (p and 255) / 255f
            if (inputType == DataType.FLOAT32) { input.putFloat(r); input.putFloat(g); input.putFloat(b) }
            else { input.put((r * 255).toInt().toByte()); input.put((g * 255).toInt().toByte()); input.put((b * 255).toInt().toByte()) }
        }
        input.rewind(); bitmap.recycle()
        val outSize = outputShape.fold(1) { a, b -> a * b }
        val output = Array(1) { FloatArray(outSize) }
        interpreter.run(input, output)
        return parse(output[0], threshold)
    }

    private fun parse(raw: FloatArray, threshold: Float): List<Pose> {
        if (raw.size < ATTR * MAX || raw.size % ATTR != 0) return emptyList()
        val n = raw.size / ATTR
        val channelsFirst = outputShape.size >= 3 && outputShape[1] == ATTR
        fun v(i: Int, a: Int): Float = if (channelsFirst) raw[a * n + i] else raw[i * ATTR + a]
        val candidates = ArrayList<Pose>()
        for (i in 0 until min(n, MAX)) {
            val cx = v(i, 0); val cy = v(i, 1); val w = v(i, 2); val h = v(i, 3)
            val score = sigmoid(v(i, 4))
            if (score < threshold) continue
            val l = ((cx - w / 2f) / INPUT).coerceIn(0f, 1f)
            val t = ((cy - h / 2f) / INPUT).coerceIn(0f, 1f)
            val r = ((cx + w / 2f) / INPUT).coerceIn(0f, 1f)
            val b = ((cy + h / 2f) / INPUT).coerceIn(0f, 1f)
            if (r <= l || b <= t) continue
            val kp = FloatArray(51)
            for (k in 0 until 17) {
                val base = 5 + k * 3
                kp[k * 3] = (v(i, base) / INPUT).coerceIn(0f, 1f)
                kp[k * 3 + 1] = (v(i, base + 1) / INPUT).coerceIn(0f, 1f)
                kp[k * 3 + 2] = sigmoid(v(i, base + 2))
            }
            candidates += Pose(android.graphics.RectF(l, t, r, b), score, kp)
        }
        candidates.sortByDescending { it.confidence }
        val result = ArrayList<Pose>()
        for (c in candidates) {
            if (result.none { iou(it.rect, c.rect) > .45f }) {
                result += c
                if (result.size == 20) break
            }
        }
        return result
    }

    private fun sigmoid(x: Float) = if (x >= 0f) 1f / (1f + exp(-x)) else { val e = exp(x); e / (1f + e) }
    private fun iou(a: android.graphics.RectF, b: android.graphics.RectF): Float {
        val l = max(a.left, b.left); val t = max(a.top, b.top); val r = min(a.right, b.right); val bot = min(a.bottom, b.bottom)
        val inter = max(0f, r - l) * max(0f, bot - t)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union > 0f) inter / union else 0f
    }

    override fun close() = interpreter.close()
}

private fun ImageProxy.toBitmap(): Bitmap? = try {
    val y = planes[0].buffer.duplicate(); val u = planes[1].buffer.duplicate(); val v = planes[2].buffer.duplicate()
    val yRow = planes[0].rowStride; val yPixel = planes[0].pixelStride
    val uRow = planes[1].rowStride; val uPixel = planes[1].pixelStride
    val vRow = planes[2].rowStride; val vPixel = planes[2].pixelStride
    val nv21 = ByteArray(width * height * 3 / 2)
    var o = 0
    val lineY = ByteArray(yRow)
    for (row in 0 until height) {
        val n = min(yRow, y.remaining()); y.get(lineY, 0, n)
        if (yPixel == 1) { System.arraycopy(lineY, 0, nv21, o, width); o += width }
        else for (col in 0 until width) nv21[o++] = lineY[col * yPixel]
    }
    val lineU = ByteArray(uRow); val lineV = ByteArray(vRow)
    for (row in 0 until height / 2) {
        val nu = min(uRow, u.remaining()); val nv = min(vRow, v.remaining()); u.get(lineU, 0, nu); v.get(lineV, 0, nv)
        for (col in 0 until width / 2) { val ui = col * uPixel; val vi = col * vPixel; nv21[o++] = lineV[vi]; nv21[o++] = lineU[ui] }
    }
    val yuv = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
    val out = java.io.ByteArrayOutputStream(); yuv.compressToJpeg(android.graphics.Rect(0, 0, width, height), 90, out)
    val decoded = android.graphics.BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size()) ?: return null
    if (imageInfo.rotationDegrees == 0) decoded else {
        val rotated = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, Matrix().apply { postRotate(imageInfo.rotationDegrees.toFloat()) }, true)
        if (rotated !== decoded) decoded.recycle(); rotated
    }
} catch (_: Throwable) { null }
