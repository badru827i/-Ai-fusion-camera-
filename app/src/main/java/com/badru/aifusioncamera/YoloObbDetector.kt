package com.badru.aifusioncamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/** Android bridge for an Ultralytics YOLO-OBB TFLite export. */
class YoloObbDetector private constructor(private val interpreter: Interpreter) {
    companion object {
        const val MODEL_NAME = "yolo11x-obb.tflite"
        private const val INPUT = 640
        private const val MAX_CANDIDATES = 8400

        fun load(context: Context): YoloObbDetector? {
            val file = File(context.filesDir, "models/$MODEL_NAME")
            if (!file.exists() || file.length() <= 0L) return null
            return try {
                FileInputStream(file).use { stream ->
                    val mapped = stream.channel.map(FileChannel.MapMode.READ_ONLY, 0L, file.length())
                    val options = Interpreter.Options()
                    options.setNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 4))
                    YoloObbDetector(Interpreter(mapped, options))
                }
            } catch (_: Throwable) {
                null
            }
        }
    }

    data class Detection(
        val rect: RectF,
        val confidence: Float,
        val angle: Float,
        val classIndex: Int
    )

    private val inputType: DataType = interpreter.getInputTensor(0).dataType()
    private val outputShape: IntArray = interpreter.getOutputTensor(0).shape()

    fun detect(proxy: ImageProxy, threshold: Float): List<Detection> {
        val source = proxy.toBitmapForObb() ?: return emptyList()
        val square = Bitmap.createBitmap(INPUT, INPUT, Bitmap.Config.ARGB_8888)
        android.graphics.Canvas(square).drawBitmap(
            source,
            null,
            Rect(0, 0, INPUT, INPUT),
            android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)
        )
        source.recycle()

        val inputBytes = if (inputType == DataType.FLOAT32) 4 else 1
        val input = ByteBuffer.allocateDirect(INPUT * INPUT * 3 * inputBytes).order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT * INPUT)
        square.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT)
        for (pixel in pixels) {
            val r = (pixel shr 16) and 255
            val g = (pixel shr 8) and 255
            val b = pixel and 255
            if (inputType == DataType.FLOAT32) {
                input.putFloat(r / 255f)
                input.putFloat(g / 255f)
                input.putFloat(b / 255f)
            } else {
                input.put(r.toByte())
                input.put(g.toByte())
                input.put(b.toByte())
            }
        }
        input.rewind()
        square.recycle()

        val outputSize = outputShape.fold(1) { acc, value -> acc * value }
        val output = Array(1) { FloatArray(outputSize) }
        return try {
            interpreter.run(input, output)
            parse(output[0], threshold)
        } catch (_: Throwable) {
            emptyList()
        }
    }

    private fun parse(raw: FloatArray, threshold: Float): List<Detection> {
        if (outputShape.size < 3) return emptyList()

        val dim1 = outputShape[1]
        val dim2 = outputShape[2]
        val channelFirst = dim1 < dim2
        val attrs = if (channelFirst) dim1 else dim2
        val count = if (channelFirst) dim2 else dim1
        if (attrs < 6 || count <= 0) return emptyList()

        // YOLO OBB export is normally [cx,cy,w,h,class scores...,angle].
        val classCount = (attrs - 5).coerceAtLeast(1)
        val angleIndex = attrs - 1
        fun value(candidate: Int, attr: Int): Float {
            return if (channelFirst) raw[attr * count + candidate]
            else raw[candidate * attrs + attr]
        }

        val detections = ArrayList<Detection>()
        for (i in 0 until min(count, MAX_CANDIDATES)) {
            val cx = value(i, 0)
            val cy = value(i, 1)
            val w = value(i, 2)
            val h = value(i, 3)
            if (!cx.isFinite() || !cy.isFinite() || !w.isFinite() || !h.isFinite() || w <= 0f || h <= 0f) continue

            var bestClass = 0
            var bestScore = 0f
            for (c in 0 until classCount) {
                val index = 4 + c
                if (index >= angleIndex) break
                val score = sigmoid(value(i, index))
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }
            if (bestScore < threshold) continue

            val left = ((cx - w / 2f) / INPUT).coerceIn(0f, 1f)
            val top = ((cy - h / 2f) / INPUT).coerceIn(0f, 1f)
            val right = ((cx + w / 2f) / INPUT).coerceIn(0f, 1f)
            val bottom = ((cy + h / 2f) / INPUT).coerceIn(0f, 1f)
            if (right <= left || bottom <= top) continue

            detections += Detection(
                rect = RectF(left, top, right, bottom),
                confidence = bestScore,
                angle = if (angleIndex < attrs) value(i, angleIndex) else 0f,
                classIndex = bestClass
            )
        }

        detections.sortByDescending { it.confidence }
        val kept = ArrayList<Detection>()
        for (candidate in detections) {
            if (kept.none { iou(it.rect, candidate.rect) > 0.45f }) {
                kept += candidate
                if (kept.size >= 50) break
            }
        }
        return kept
    }

    private fun sigmoid(x: Float): Float {
        return if (x >= 0f) 1f / (1f + exp(-x)) else {
            val e = exp(x)
            e / (1f + e)
        }
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union > 0f) intersection / union else 0f
    }

    fun close() {
        interpreter.close()
    }
}

private fun ImageProxy.toBitmapForObb(): Bitmap? = try {
    val y = planes[0].buffer.duplicate()
    val u = planes[1].buffer.duplicate()
    val v = planes[2].buffer.duplicate()
    val yRowStride = planes[0].rowStride
    val yPixelStride = planes[0].pixelStride
    val uRowStride = planes[1].rowStride
    val uPixelStride = planes[1].pixelStride
    val vRowStride = planes[2].rowStride
    val vPixelStride = planes[2].pixelStride

    val nv21 = ByteArray(width * height * 3 / 2)
    var offset = 0
    val yRow = ByteArray(yRowStride)
    for (row in 0 until height) {
        val n = min(yRowStride, y.remaining())
        y.get(yRow, 0, n)
        if (yPixelStride == 1) {
            System.arraycopy(yRow, 0, nv21, offset, width)
            offset += width
        } else {
            for (col in 0 until width) nv21[offset++] = yRow[col * yPixelStride]
        }
    }

    val uRow = ByteArray(uRowStride)
    val vRow = ByteArray(vRowStride)
    for (row in 0 until height / 2) {
        val un = min(uRowStride, u.remaining())
        val vn = min(vRowStride, v.remaining())
        u.get(uRow, 0, un)
        v.get(vRow, 0, vn)
        for (col in 0 until width / 2) {
            val ui = col * uPixelStride
            val vi = col * vPixelStride
            if (ui < un && vi < vn) {
                nv21[offset++] = vRow[vi]
                nv21[offset++] = uRow[ui]
            }
        }
    }

    val yuv = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
    val stream = ByteArrayOutputStream()
    yuv.compressToJpeg(Rect(0, 0, width, height), 90, stream)
    val decoded = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size()) ?: return null
    val degrees = imageInfo.rotationDegrees
    if (degrees == 0) decoded else {
        val rotated = Bitmap.createBitmap(
            decoded,
            0,
            0,
            decoded.width,
            decoded.height,
            Matrix().apply { postRotate(degrees.toFloat()) },
            true
        )
        if (rotated !== decoded) decoded.recycle()
        rotated
    }
} catch (_: Throwable) {
    null
}
