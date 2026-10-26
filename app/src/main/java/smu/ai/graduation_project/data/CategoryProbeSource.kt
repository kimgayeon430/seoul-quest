package smu.ai.graduation_project.data

import android.content.Context
import android.util.Log
import org.json.JSONObject
import smu.ai.graduation_project.domain.CategoryProbe

/**
 * `assets/clip_probe.json` (= `ml/eval/eval_clip_probe.py --export` 산출물) 을 읽어 [CategoryProbe.Model] 로 만든다.
 *
 * 파일이 없거나 형식이 맞지 않으면 null 을 돌려주고, 사진 인증은 MobileViT 점수만으로 판정한다.
 */
object CategoryProbeSource {

    private const val TAG = "CategoryProbeSource"

    fun load(context: Context, asset: String = "clip_probe.json"): CategoryProbe.Model? = runCatching {
        val json = JSONObject(context.assets.open(asset).use { it.reader().readText() })
        val labels = json.getJSONArray("labels").let { a -> (0 until a.length()).map { a.getString(it) } }
        val w = json.getJSONArray("weights")
        val b = json.getJSONArray("bias")
        val keep = json.optJSONArray("keep_labels")?.let { a -> (0 until a.length()).map { a.getString(it) }.toSet() }
        CategoryProbe.Model(
            labels = labels,
            weights = (0 until w.length()).map { i -> w.getJSONArray(i).let { r -> DoubleArray(r.length()) { r.getDouble(it) } } },
            bias = DoubleArray(b.length()) { b.getDouble(it) },
            blend = json.getDouble("blend"),
            keepLabels = keep ?: setOf(smu.ai.graduation_project.domain.PhotoVerification.INVALID_LABEL),
            version = json.optString("version", "clip-probe"),
        )
    }.onFailure { Log.i(TAG, "clip probe 미탑재/무효 → MobileViT 점수만 사용: ${it.message}") }.getOrNull()
}
