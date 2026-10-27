package smu.ai.graduation_project.domain

import kotlin.math.exp

/**
 * CLIP 이미지 임베딩 위의 선형 분류기(linear probe)로 카테고리 점수를 보정한다. 순수 Kotlin.
 *
 * 앱은 대표 이미지 유사도를 위해 촬영본의 CLIP 임베딩을 이미 계산한다([PhotoVerification.Classification.embedding]).
 * 그 임베딩에 5클래스 로지스틱 회귀(가중치 라벨×512 + 절편, `assets/clip_probe.json`)를 얹어 두 번째 분류기를
 * 얻고, MobileViT 점수와 섞는다: `점수 = (1 − blend)·MobileViT + blend·probe`.
 *
 * [Model.keepLabels](기본: 무효)는 섞지 않고 MobileViT 점수를 그대로 둔다. 무효 게이트는 화면 재촬영 같은
 * 스푸핑을 막는 첫 관문이라, 그 학습 데이터로 직접 학습한 MobileViT 판단을 유지한다.
 *
 * 근거(`ml/eval/eval_clip_probe.py`, test 893장, C·blend 는 validation 에서 선택): macro-F1 0.811 → 0.828,
 * 정상 사진 오거절률 0.146 → 0.108 (McNemar p = 0.006), 무효 recall 0.932 유지. 앱이 쓰는 int8 임베더로도
 * 같은 수준(`eval_int8_embedder.py`, macro-F1 0.830).
 */
object CategoryProbe {

    class Model(
        val labels: List<String>,
        /** labels 순서의 행. 각 행은 임베딩 차원(512) 길이. */
        val weights: List<DoubleArray>,
        val bias: DoubleArray,
        val blend: Double,
        val keepLabels: Set<String> = setOf(PhotoVerification.INVALID_LABEL),
        val version: String = "clip-probe",
    ) {
        init {
            require(labels.size == weights.size && labels.size == bias.size) { "labels/weights/bias 크기 불일치" }
            require(blend in 0.0..1.0) { "blend 는 0..1 범위여야 합니다." }
        }
    }

    /** 임베딩 → 라벨별 softmax 확률. 차원이 맞지 않으면 null. */
    fun probsOrNull(model: Model, embedding: FloatArray?): Map<String, Double>? {
        if (embedding == null || model.weights.any { it.size != embedding.size }) return null
        val logits = DoubleArray(model.labels.size) { i ->
            val w = model.weights[i]
            var z = model.bias[i]
            for (k in embedding.indices) z += w[k] * embedding[k]
            z
        }
        val maxZ = logits.max()
        val exps = logits.map { exp(it - maxZ) }
        val sum = exps.sum()
        return model.labels.indices.associate { model.labels[it] to exps[it] / sum }
    }

    /**
     * MobileViT 점수에 probe 를 섞은 분류 결과. 임베딩이 없거나 probe 를 계산할 수 없으면 입력을 그대로 돌려준다.
     */
    fun apply(model: Model?, classification: PhotoVerification.Classification): PhotoVerification.Classification {
        if (model == null) return classification
        val probe = probsOrNull(model, classification.embedding) ?: return classification
        val mixed = classification.scores.mapValues { (label, score) ->
            val p = probe[label]
            if (p == null || label in model.keepLabels) score else (1 - model.blend) * score + model.blend * p
        }
        return classification.copy(scores = mixed)
    }
}
