package com.badru.aifusioncamera

import android.content.Context
import android.net.Uri
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.security.MessageDigest

/**
 * Persistent local model registry.
 *
 * Rules:
 * - .tflite models are validated before becoming ACTIVE.
 * - .pt models can be stored as source checkpoints but are marked NEEDS_CONVERSION;
 *   TensorFlow Lite will never pretend that a PyTorch checkpoint is executable.
 * - The last validated model is persisted and automatically restored on app start.
 */
object ModelManager {
    private const val DIR = "models"
    private const val MANIFEST = "model_manifest.json"
    private const val PREFS = "model_manager"
    private const val ACTIVE_ID = "active_model_id"

    enum class ModelKind { POSE, OBB, UNKNOWN }
    enum class ModelState { ACTIVE, STORED, NEEDS_CONVERSION, INVALID }

    data class ModelInfo(
        val id: String,
        val fileName: String,
        val localPath: String,
        val kind: ModelKind,
        val state: ModelState,
        val sizeBytes: Long,
        val sha256: String,
        val inputShape: List<Int> = emptyList(),
        val inputType: String = "",
        val outputShapes: List<List<Int>> = emptyList(),
        val outputTypes: List<String> = emptyList(),
        val message: String = ""
    )

    data class ValidationResult(
        val ok: Boolean,
        val message: String,
        val inputShape: List<Int> = emptyList(),
        val inputType: String = "",
        val outputShapes: List<List<Int>> = emptyList(),
        val outputTypes: List<String> = emptyList()
    )

    private fun modelDir(context: Context): File = File(context.filesDir, DIR).apply { mkdirs() }

    private fun manifestFile(context: Context): File = File(modelDir(context), MANIFEST)

    private fun prefs(context: Context) = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun kindForFileName(fileName: String): ModelKind {
        val n = fileName.lowercase()
        return when {
            "pose" in n -> ModelKind.POSE
            "obb" in n || "oriented" in n -> ModelKind.OBB
            else -> ModelKind.UNKNOWN
        }
    }

    fun importFromUri(context: Context, uri: Uri, originalName: String?): ModelInfo {
        val name = (originalName ?: "model.tflite").ifBlank { "model.tflite" }
        val safeName = name.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val target = File(modelDir(context), safeName)
        context.contentResolver.openInputStream(uri)?.use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
        } ?: return ModelInfo(
            id = safeName,
            fileName = safeName,
            localPath = target.absolutePath,
            kind = kindForFileName(safeName),
            state = ModelState.INVALID,
            sizeBytes = 0L,
            sha256 = "",
            message = "Unable to read model file"
        )
        return register(context, target)
    }

    fun register(context: Context, file: File): ModelInfo {
        if (!file.exists() || file.length() <= 0L) {
            return buildInfo(file, ModelState.INVALID, "Model file is empty or missing")
        }
        val kind = kindForFileName(file.name)
        if (file.extension.equals("pt", ignoreCase = true)) {
            val info = buildInfo(file, ModelState.NEEDS_CONVERSION, "PyTorch .pt checkpoint stored. Convert to .tflite/LiteRT before Android inference.")
            saveManifestEntry(context, info)
            return info
        }
        if (!file.extension.equals("tflite", ignoreCase = true)) {
            val info = buildInfo(file, ModelState.INVALID, "Unsupported model format: .${file.extension}")
            saveManifestEntry(context, info)
            return info
        }
        val validation = validateTflite(file)
        val state = if (validation.ok) ModelState.ACTIVE else ModelState.INVALID
        val info = buildInfo(
            file,
            state,
            validation.message,
            validation.inputShape,
            validation.inputType,
            validation.outputShapes,
            validation.outputTypes
        )
        saveManifestEntry(context, info)
        if (state == ModelState.ACTIVE) prefs(context).edit().putString(ACTIVE_ID, info.id).apply()
        return info
    }

    fun restoreActive(context: Context): ModelInfo? {
        val id = prefs(context).getString(ACTIVE_ID, null) ?: return null
        val entries = readManifest(context)
        val existing = entries.firstOrNull { it.id == id } ?: return null
        val file = File(existing.localPath)
        if (!file.exists() || file.length() == 0L) return null
        if (!file.extension.equals("tflite", ignoreCase = true)) return existing
        val validation = validateTflite(file)
        if (!validation.ok) {
            val invalid = existing.copy(state = ModelState.INVALID, message = validation.message)
            saveManifestEntry(context, invalid)
            prefs(context).edit().remove(ACTIVE_ID).apply()
            return invalid
        }
        val restored = existing.copy(
            state = ModelState.ACTIVE,
            inputShape = validation.inputShape,
            inputType = validation.inputType,
            outputShapes = validation.outputShapes,
            outputTypes = validation.outputTypes,
            message = validation.message
        )
        saveManifestEntry(context, restored)
        return restored
    }

    fun activeFile(context: Context): File? {
        val active = restoreActive(context) ?: return null
        if (active.state != ModelState.ACTIVE) return null
        val file = File(active.localPath)
        return if (file.exists()) file else null
    }

    fun listModels(context: Context): List<ModelInfo> = readManifest(context)

    fun clearModel(context: Context, id: String) {
        val entries = readManifest(context).filterNot { it.id == id }
        val target = File(modelDir(context), id)
        if (target.exists()) target.delete()
        writeManifest(context, entries)
        if (prefs(context).getString(ACTIVE_ID, null) == id) prefs(context).edit().remove(ACTIVE_ID).apply()
    }

    fun validateTflite(file: File): ValidationResult {
        return try {
            val stream = FileInputStream(file)
            val mapped = stream.channel.map(FileChannel.MapMode.READ_ONLY, 0L, file.length())
            stream.close()
            Interpreter(mapped, Interpreter.Options().apply {
                setNumThreads(2)
            }).use { interpreter ->
                if (interpreter.inputTensorCount < 1 || interpreter.outputTensorCount < 1) {
                    return ValidationResult(false, "Model has no input/output tensors")
                }
                val input = interpreter.getInputTensor(0)
                val inputShape = input.shape().toList()
                val outputs = (0 until interpreter.outputTensorCount).map { interpreter.getOutputTensor(it) }
                val outputShapes = outputs.map { it.shape().toList() }
                val outputTypes = outputs.map { it.dataType().name }

                if (inputShape.size != 4) {
                    return ValidationResult(false, "Unsupported input rank ${inputShape.size}; expected image tensor rank 4", inputShape, input.dataType().name, outputShapes, outputTypes)
                }
                if (inputShape[0] != 1) {
                    return ValidationResult(false, "Only batch=1 is supported; model reports batch=${inputShape[0]}", inputShape, input.dataType().name, outputShapes, outputTypes)
                }
                val h = inputShape.getOrNull(1) ?: 0
                val w = inputShape.getOrNull(2) ?: 0
                val c = inputShape.getOrNull(3) ?: 0
                if (h <= 0 || w <= 0 || c != 3) {
                    return ValidationResult(false, "Unsupported input shape ${inputShape}; expected [1,H,W,3]", inputShape, input.dataType().name, outputShapes, outputTypes)
                }

                // Dry-run one inference. This catches corrupted/incompatible tensors before activation.
                val inputBuffer = ByteBuffer.allocateDirect(input.numBytes()).order(ByteOrder.nativeOrder())
                val outputMap = HashMap<Int, Any>()
                for (i in outputs.indices) {
                    outputMap[i] = ByteBuffer.allocateDirect(outputs[i].numBytes()).order(ByteOrder.nativeOrder())
                }
                interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
                ValidationResult(
                    true,
                    "Model validated and dry-run inference passed",
                    inputShape,
                    input.dataType().name,
                    outputShapes,
                    outputTypes
                )
            }
        } catch (t: Throwable) {
            ValidationResult(false, "Model validation failed: ${t.message ?: t.javaClass.simpleName}")
        }
    }

    private fun buildInfo(
        file: File,
        state: ModelState,
        message: String,
        inputShape: List<Int> = emptyList(),
        inputType: String = "",
        outputShapes: List<List<Int>> = emptyList(),
        outputTypes: List<String> = emptyList()
    ): ModelInfo = ModelInfo(
        id = file.name,
        fileName = file.name,
        localPath = file.absolutePath,
        kind = kindForFileName(file.name),
        state = state,
        sizeBytes = file.length(),
        sha256 = sha256(file),
        inputShape = inputShape,
        inputType = inputType,
        outputShapes = outputShapes,
        outputTypes = outputTypes,
        message = message
    )

    private fun sha256(file: File): String = try {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(8192)
            while (true) {
                val n = input.read(buffer)
                if (n <= 0) break
                digest.update(buffer, 0, n)
            }
        }
        digest.digest().joinToString("") { "%02x".format(it) }
    } catch (_: Exception) { "" }

    private fun readManifest(context: Context): MutableList<ModelInfo> {
        val file = manifestFile(context)
        if (!file.exists()) return mutableListOf()
        return try {
            val array = JSONArray(file.readText())
            MutableList(array.length()) { i -> fromJson(array.getJSONObject(i)) }
        } catch (_: Exception) {
            mutableListOf()
        }
    }

    private fun saveManifestEntry(context: Context, info: ModelInfo) {
        val entries = readManifest(context).filterNot { it.id == info.id }.toMutableList()
        entries.add(info)
        writeManifest(context, entries)
    }

    private fun writeManifest(context: Context, entries: List<ModelInfo>) {
        val array = JSONArray()
        entries.forEach { array.put(toJson(it)) }
        manifestFile(context).writeText(array.toString())
    }

    private fun toJson(info: ModelInfo): JSONObject = JSONObject().apply {
        put("id", info.id)
        put("fileName", info.fileName)
        put("localPath", info.localPath)
        put("kind", info.kind.name)
        put("state", info.state.name)
        put("sizeBytes", info.sizeBytes)
        put("sha256", info.sha256)
        put("inputShape", JSONArray(info.inputShape))
        put("inputType", info.inputType)
        put("outputShapes", JSONArray(info.outputShapes.map { JSONArray(it) }))
        put("outputTypes", JSONArray(info.outputTypes))
        put("message", info.message)
    }

    private fun fromJson(o: JSONObject): ModelInfo = ModelInfo(
        id = o.optString("id"),
        fileName = o.optString("fileName"),
        localPath = o.optString("localPath"),
        kind = runCatching { ModelKind.valueOf(o.optString("kind", ModelKind.UNKNOWN.name)) }.getOrDefault(ModelKind.UNKNOWN),
        state = runCatching { ModelState.valueOf(o.optString("state", ModelState.STORED.name)) }.getOrDefault(ModelState.STORED),
        sizeBytes = o.optLong("sizeBytes", 0L),
        sha256 = o.optString("sha256"),
        inputShape = o.optJSONArray("inputShape")?.let { a -> List(a.length()) { i -> a.optInt(i) } } ?: emptyList(),
        inputType = o.optString("inputType"),
        outputShapes = o.optJSONArray("outputShapes")?.let { a -> List(a.length()) { i ->
            val row = a.optJSONArray(i) ?: JSONArray()
            List(row.length()) { j -> row.optInt(j) }
        } } ?: emptyList(),
        outputTypes = o.optJSONArray("outputTypes")?.let { a -> List(a.length()) { i -> a.optString(i) } } ?: emptyList(),
        message = o.optString("message")
    )
}
