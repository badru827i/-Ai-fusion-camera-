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

/** Generic Ultralytics YOLO-OBB TFLite bridge.
 * Expected raw export layout is [1, 5+numClasses, N] or [1, N, 5+numClasses].
 * The last scalar is interpreted as rotation angle when present.
 */
class YoloObbDetector private constructor(private val interpreter: Interpreter) : AutoCloseable {
    companion object {
        const val MODEL_NAME = "yolo-obb.tflite"
        private const val INPUT = 640
        private const val MAX = 8400

        fun load(context: Context): YoloObbDetector? = try {
            val imported = File(context.filesDir, "models/$MODEL_NAME")
            if (!imported.exists() || imported.length() == 0L) return null
            val stream = FileInputStream(imported)
            val mapped = stream.channel.map(FileChannel.MapMode.READ_ONLY, 0, imported.length())
            stream.close()
            val options = Interpreter.Options().apply {
                setNumThreads(max(2, min(4, Runtime.getRuntime().availableProcessors())))
            }
            YoloObbDetector(Interpreter(mapped, options))
        } catch (_: Throwable) {
            null
        }
    }

    data class Detection(
        val rect: android.graphics.RectF,
        val confidence: Float,
        val angle: Float,
        val classIndex: Int
    )

    private val inputType = interpreter.getInputTensor(0).dataType()
    private val outputShape = interpreter.getOutputTensor(0).shape()
    private val outputSize = outputShape.fold(1) { a, b -> a * b }

    fun detect(proxy: ImageProxy, threshold: Float): List<Detection> {
        val source = proxy.toBitmapForObb() ?: return emptyList()
        val bitmap = Bitmap.createBitmap(INPUT, INPUT, Bitmap.Config.ARGB_8888)
        android.graphics.Canvas(bitmap).drawBitmap(
            source, null, android.graphics.Rect(0, 0, INPUT, INPUT),
            android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)
        )
        source.recycle()
        val bytes = if (inputType == DataType.FLOAT32) 4 else 1
        val input = ByteBuffer.allocateDirect(INPUT * INPUT * 3 * bytes).order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT * INPUT)
        bitmap.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT)
        for (p in pixels) {
            val r = ((p shr 16) and 255) / 255f
            val g = ((p shr 8) and 255) / 255f
            val b = (p and 255) / 255f
            if (inputType == DataType.FLOAT32) {
                input.putFloat(r); input.putFloat(g); input.putFloat(b)
            } else {
                input.put((r * 255).toInt().toByte()); input.put((g * 255).toInt().toByte()); input.put((b * 255).toInt().toByte())
            }
        }
        input.rewind()
        bitmap.recycle()
        val output = Array(1) { FloatArray(outputSize) }
        interpreter.run(input, output)
        return parse(output[0], threshold)
    }

    private fun parse(raw: FloatArray, threshold: Float): List<Detection> {
        if (outputShape.size < 3) return emptyList()
        val attrs = outputShape[outputShape.size - 2]
        val candidates = outputShape.last()
        val channelFirst = attrs < candidates
        val attrCount = if (channelFirst) outputShape[1] else outputShape[2]
        val n = if (channelFirst) outputShape[2] else outputShape[1]
        if (attrCount < 6 || n <= 0) return emptyList()
        val classCount = attrCount - 5
        fun value(i: Int, a: Int): Float = if (channelFirst) raw[a * n + i] else raw[i * attrCount + a]

        val out = ArrayList<Detection>()
        for (i in 0 until min(n, MAX)) {
            val cx = value(i, 0); val cy = value(i, 1); val w = value(i, 2); val h = value(i, 3)
            var bestClass = -1
            var bestScore = 0f
            for (c in 0 until classCount) {
                val score = sigmoid(value(i, 4 + c))
                if (score > bestScore) { bestScore = score; bestClass = c }
            }
            if (bestClass < 0 || bestScore < threshold) continue
            val l = ((cx - w / 2f) / INPUT).coerceIn(0f, 1f)
            val t = ((cy - h / 2f) / INPUT).coerceIn(0f, 1f)
            val r = ((cx + w / 2f) / INPUT).coerceIn(0f, 1f)
            val b = ((cy + h / 2f) / INPUT).coerceIn(0f, 1f)
            if (r <= l || b <= t) continue
            val angle = if (attrCount > 5 + classCount) value(i, 4 + classCount) else 0f
            out += Detection(android.graphics.RectF(l, t, r, b), bestScore, angle, bestClass)
        }
        out.sortByDescending { it.confidence }
        val kept = ArrayList<Detection>()
        for (d in out) {
            if (kept.none { iou(it.rect, d.rect) > .45f }) {
                kept += d
                if (kept.size >= 50) break
            }
        }
        return kept
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

private fun ImageProxy.toBitmapForObb(): Bitmap? = try {
    val y = planes[0].buffer.duplicate(); val u = planes[1].buffer.duplicate(); val v = planes[2].buffer.duplicate()
    val yRow = planes[0].rowStride; val yPixel = planes[0].pixelStride
    val uRow = planes[1].rowStride; val uPixel = planes[1].pixelStride
    val vRow = planes[2].rowStride; val vPixel = planes[2].pixelStride
    val nv21 = ByteArray(width * height * 3 / 2)
    var o = 0
    val ly = ByteArray(yRow)
    for (row in 0 until height) {
        val n = min(yRow, y.remaining()); y.get(ly, 0, n)
        if (yPixel == 1) { System.arraycopy(ly, 0, nv21, o, width); o += width }
        else for (col in 0 until width) nv21[o++] = ly[col * yPixel]
    }
    val lu = ByteArray(uRow); val lv = ByteArray(vRow)
    for (row in 0 until height / 2) {
        val nu = min(uRow, u.remaining()); val nv = min(vRow, v.remaining())
        u.get(lu, 0, nu); v.get(lv, 0, nv)
        for (col in 0 until width / 2) {
            val ui = col * uPixel; val vi = col * vPixel
            if (ui < nu && vi < nv) { nv21[o++] = lv[vi]; nv21[o++] = lu[ui] }
        }
    }
    val yuv = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
    val bytes = java.io.ByteArrayOutputStream(); yuv.compressToJpeg(android.graphics.Rect(0, 0, width, height), 90, bytes)
    val decoded = android.graphics.BitmapFactory.decodeByteArray(bytes.toByteArray(), 0, bytes.size()) ?: return null
    if (imageInfo.rotationDegrees == 0) decoded else {
        val rotated = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, Matrix().apply { postRotate(imageInfo.rotationDegrees.toFloat()) }, true)
        if (rotated !== decoded) decoded.recycle(); rotated
    }
} catch (_: Throwable) { null }
