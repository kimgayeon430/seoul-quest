package smu.ai.graduation_project.domain

import kotlin.math.exp
import kotlin.math.sqrt

/**
 * 사진 인증(2단계) 판정 규칙. 순수 Kotlin (모델·Android·Firebase 비의존).
 *
 * 온디바이스 비전 모델이 촬영본을 카테고리별 점수로 분류하면([Classification]),
 * 그 점수와 미션이 기대하는 카테고리를 비교해 세 갈래로 판정한다.
 *
 *  - [Verdict.PASS]         : 자동 통과. 바로 완료 처리.
 *  - [Verdict.NEEDS_REVIEW] : 애매함. 미션은 완료하되 관리자 검수 큐에 올린다.
 *  - [Verdict.REJECT]       : 부적합. 업로드하지 않고 재촬영을 요청한다.
 *
 * 임계값은 [PhotoVerificationConfig] 에 모여 있어 오프라인 평가 후 조정할 수 있다.
 *
 * ### 참조 이미지 유사도 (선택)
 * 카테고리 분류만으로는 "미션 *유형* 에 맞는 사진인가"만 볼 수 있고 "*이* 미션의 대상을
 * 찍었는가"는 못 본다(보고서 6.7). 미션 대표 이미지의 임베딩([referenceEmbeddings])이
 * 주어지면, 촬영본 임베딩([Classification.embedding])과의 코사인 유사도를 **보조 신호**로
 * 결합한다. 유사도만으로 통과/거절을 뒤집지 않고, 한 단계씩만 조정한다(6.7.5).
 *
 * 참조 이미지가 여러 장이면(각도·조명·계절이 다른 사진) [PhotoEmbedding.maxCosineOrNull] 로
 * **최대 유사도**를 취한다 — 한 장만 닮아도 같은 대상으로 인정한다(6.7.6/6.7.8, 오프라인
 * 프로토타입에서 Youden J 0.33 → 0.47 로 분리력 개선 확인).
 *
 * ### 미션 설명 텍스트 일치도 (선택)
 * CLIP 은 이미지와 텍스트를 같은 임베딩 공간에 둔다. 미션 문구("a photo of tteokbokki at
 * Gwangjang Market" 등)의 텍스트 임베딩([TextAnchors.positives])과, 다른 미션·무효 문구의 텍스트
 * 임베딩([TextAnchors.negatives])을 함께 두고 촬영본이 **이 미션 문구에 얼마나 더 가까운지**를
 * softmax 확률([PhotoEmbedding.textMatchProbOrNull])로 계산한다. 대표 이미지가 한 장뿐이라
 * 각도·구도가 달라 유사도가 낮게 나오는 경우를 보완하고, 대표 이미지가 없는 미션에도 쓸 수 있다.
 * 대표 이미지 유사도와 똑같이 보조 신호로만 쓰며, 둘 중 하나라도 근거가 되면 인정한다.
 */
object PhotoVerification {

    /** 무효(셀카·스크린샷·무관 실내 등) 클래스 라벨. 모델 학습 라벨과 일치해야 한다. */
    const val INVALID_LABEL = "무효"

    enum class Verdict { PASS, NEEDS_REVIEW, REJECT }

    /**
     * [Verdict.REJECT] 일 때 사용자에게 보여줄 안내 문구를 고르기 위한 원인 분류.
     * 화면 표시 문구(다국어)는 UI 레이어(`strings.xml`)에서 맡고, 여기서는 원인만 분류한다.
     */
    enum class RejectReasonCode { INVALID_SUBJECT, CATEGORY_MISMATCH, LOW_CONFIDENCE }

    /** 모델이 낸 라벨별 점수. 합이 1일 필요는 없다. */
    data class Classification(
        val scores: Map<String, Double>,
        /**
         * 촬영본의 이미지 임베딩(L2 정규화 권장). 임베더 모델이 없으면 null.
         * [referenceEmbedding] 과 코사인 유사도를 계산하는 데만 쓴다.
         */
        val embedding: FloatArray? = null,
    ) {
        fun scoreOf(label: String): Double = scores[label] ?: 0.0
        val topLabel: String? get() = scores.maxByOrNull { it.value }?.key

        // data class + FloatArray: equals/hashCode 를 배열 내용 기준으로 직접 구현한다.
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Classification) return false
            return scores == other.scores && embedding.contentEqualsNullable(other.embedding)
        }

        override fun hashCode(): Int =
            31 * scores.hashCode() + (embedding?.contentHashCode() ?: 0)
    }

    /**
     * 미션 설명 텍스트의 CLIP 텍스트 임베딩. `ml/embed_mission_texts.py` 로 사전계산해
     * `missions/{id}.photoTextEmbeddings` / `photoTextNegatives` 에 저장한다.
     *
     * @property positives  이 미션을 묘사하는 문구들(제목·설명·관리자 지정 프롬프트)의 임베딩.
     * @property negatives  대조 문구(다른 미션 문구, 셀카·스크린샷 등 무효 문구)의 임베딩.
     * @property logitScale CLIP 의 logit scale(ViT-B/32 는 100). 코사인 유사도에 곱해 softmax 에 넣는다.
     */
    class TextAnchors(
        val positives: List<FloatArray>,
        val negatives: List<FloatArray>,
        val logitScale: Double = DEFAULT_LOGIT_SCALE,
    ) {
        val isUsable: Boolean get() = positives.isNotEmpty() && negatives.isNotEmpty()

        companion object {
            const val DEFAULT_LOGIT_SCALE = 100.0
        }
    }

    data class Result(
        val verdict: Verdict,
        /** 미션 카테고리에 대한 모델 점수. */
        val matchScore: Double,
        /** 무효 라벨 점수. */
        val invalidScore: Double,
        /** 참조 이미지와의 코사인 유사도. 계산하지 못했으면 null. */
        val similarity: Double?,
        /** 사용자/관리자에게 보여줄 사유. */
        val reason: String,
        /** [Verdict.REJECT] 일 때만 값이 있다. */
        val rejectReasonCode: RejectReasonCode? = null,
        /** 미션 설명 텍스트 일치 확률(0..1, [PhotoEmbedding.textMatchProbOrNull]). 계산하지 못했으면 null. */
        val textMatch: Double? = null,
    )

    /**
     * @param missionCategory     미션이 기대하는 카테고리 (예: `"맛집"`)
     * @param classification      모델 분류 결과. 모델을 못 불러왔으면 null.
     * @param referenceEmbeddings 미션 대표 이미지(들)의 임베딩. 비어 있으면(대부분의 기존 미션) 유사도 결합을 건너뛴다.
     *                            여러 장이면 최대 유사도를 쓴다.
     * @param textAnchors         미션 설명 텍스트 임베딩. null 이면(대부분의 기존 미션) 텍스트 일치도 결합을 건너뛴다.
     */
    fun verify(
        missionCategory: String,
        classification: Classification?,
        referenceEmbeddings: List<FloatArray> = emptyList(),
        config: PhotoVerificationConfig = PhotoVerificationConfig.DEFAULT,
        textAnchors: TextAnchors? = null,
    ): Result {
        if (classification == null) {
            return if (config.passWhenModelUnavailable) {
                Result(Verdict.PASS, 0.0, 0.0, null, "자동 인증을 건너뛰고 통과했습니다.")
            } else {
                Result(Verdict.NEEDS_REVIEW, 0.0, 0.0, null,
                    "자동 인증을 할 수 없어 관리자 확인 후 포인트가 지급돼요.")
            }
        }

        val match = classification.scoreOf(missionCategory)
        val invalid = classification.scoreOf(INVALID_LABEL)

        // (1) 무효는 유사도보다 먼저 무조건 차단한다. 참조 이미지를 화면에 띄워 재촬영하는
        //     스푸핑은 유사도가 최대에 가깝게 나오므로, 유사도로는 막을 수 없다(6.7.5-가).
        if (invalid >= config.invalidRejectThreshold) {
            return Result(Verdict.REJECT, match, invalid, null,
                "사진이 미션과 무관해 보여요. 미션 장소·대상을 촬영해 주세요.",
                RejectReasonCode.INVALID_SUBJECT)
        }

        // (2) 카테고리 점수만으로 1차 판정.
        val base = when {
            match < config.hardRejectThreshold -> Verdict.REJECT
            match >= config.autoPassThreshold -> Verdict.PASS
            else -> Verdict.NEEDS_REVIEW
        }
        val baseReasonCode = if (base == Verdict.REJECT) {
            rejectReasonCode(classification, missionCategory, match)
        } else null

        val similarity = PhotoEmbedding.maxCosineOrNull(classification.embedding, referenceEmbeddings)
        val textMatch = PhotoEmbedding.textMatchProbOrNull(classification.embedding, textAnchors)
        if (similarity == null && textMatch == null) {
            return Result(base, match, invalid, null, reasonFor(base, missionCategory), baseReasonCode)
        }

        // (3) 대표 이미지 유사도·미션 텍스트 일치도를 보조 신호로 결합한다. 한 단계씩만 조정한다(6.7.5).
        //     둘 중 계산된 신호만 쓰고, 하나라도 "닮았다"고 하면 그쪽을 근거로 인정한다.
        val simAtLeast = { t: Double -> similarity != null && similarity >= t }
        val textAtLeast = { t: Double -> textMatch != null && textMatch >= t }
        // 구제 근거: 대표 이미지와 어느 정도 닮았거나, 미션 문구와 확실히 맞음.
        val rescue = simAtLeast(config.similarityRescueThreshold) || textAtLeast(config.textMatchPassThreshold)
        // 통과 근거: 대표 이미지와 매우 닮았거나, 미션 문구와 확실히 맞음.
        val support = simAtLeast(config.similaritySuspectThreshold) || textAtLeast(config.textMatchPassThreshold)
        // 반대 근거: 어느 신호도 이 미션의 대상이라고 볼 만큼 닮지 않음.
        val against = !simAtLeast(config.similaritySuspectThreshold) && !textAtLeast(config.textMatchSuspectThreshold)

        val combined = when {
            // 카테고리 점수가 낮아도 미션 대상과 충분히 닮았으면 즉시 거절하지 않고 검수로 구제.
            base == Verdict.REJECT && rescue -> Verdict.NEEDS_REVIEW
            // 애매한데 미션 대상과 매우 닮았으면 통과.
            base == Verdict.NEEDS_REVIEW && support -> Verdict.PASS
            // 유형은 맞는데 미션 대상과 안 닮음 → 같은 유형의 다른 대상 촬영 의심, 검수로.
            base == Verdict.PASS && against -> Verdict.NEEDS_REVIEW
            else -> base
        }
        // combined 가 REJECT 로 남는 경우는 base 가 이미 REJECT 였고 구제도 안 된 경우뿐이다.
        val combinedReasonCode = if (combined == Verdict.REJECT) baseReasonCode else null

        return Result(combined, match, invalid, similarity,
            reasonForCombined(base, combined, missionCategory), combinedReasonCode, textMatch)
    }

    /**
     * REJECT 원인을 분류한다. 무효 판정([INVALID_LABEL])은 [verify] 에서 먼저 걸러지므로 여기선
     * "다른 카테고리로 더 강하게 분류됐는가"(유형 불일치) vs "전반적으로 애매함"(저신뢰) 만 가른다.
     */
    private fun rejectReasonCode(
        classification: Classification,
        missionCategory: String,
        matchScore: Double
    ): RejectReasonCode {
        val topLabel = classification.topLabel
        return if (topLabel != null && topLabel != missionCategory && topLabel != INVALID_LABEL &&
            classification.scoreOf(topLabel) > matchScore
        ) {
            RejectReasonCode.CATEGORY_MISMATCH
        } else {
            RejectReasonCode.LOW_CONFIDENCE
        }
    }

    private fun reasonFor(verdict: Verdict, category: String): String = when (verdict) {
        Verdict.REJECT ->
            "사진에서 '$category' 미션 요소를 찾지 못했어요. 다시 촬영해 주세요."
        Verdict.PASS -> "사진 인증 완료"
        Verdict.NEEDS_REVIEW ->
            "자동 인증이 애매해 관리자 확인 후 포인트가 지급돼요."
    }

    private fun reasonForCombined(
        base: Verdict,
        combined: Verdict,
        category: String,
    ): String = when {
        base == Verdict.REJECT && combined == Verdict.NEEDS_REVIEW ->
            "'$category' 요소는 약하지만 미션 대상과 닮아 관리자 확인 후 지급돼요."
        base == Verdict.NEEDS_REVIEW && combined == Verdict.PASS ->
            "사진 인증 완료"
        base == Verdict.PASS && combined == Verdict.NEEDS_REVIEW ->
            "미션 대표 장소·대상과 달라 보여요. 관리자 확인 후 포인트가 지급돼요."
        else -> reasonFor(combined, category)
    }
}

/** 이미지 임베딩 유사도 계산. 순수 Kotlin. */
object PhotoEmbedding {

    /**
     * 두 벡터의 코사인 유사도(-1..1). 정규화 여부와 무관하게 각 노름으로 나눈다.
     * 한쪽이 null 이거나 길이가 다르거나 노름이 0이면 null.
     */
    fun cosineOrNull(a: FloatArray?, b: FloatArray?): Double? {
        if (a == null || b == null || a.size != b.size || a.isEmpty()) return null
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            val x = a[i].toDouble()
            val y = b[i].toDouble()
            dot += x * y
            na += x * x
            nb += y * y
        }
        if (na <= 0.0 || nb <= 0.0) return null
        return dot / (sqrt(na) * sqrt(nb))
    }

    /**
     * 참조 이미지가 여러 장일 때 **최대 유사도**를 취한다(6.7.8). 각도·조명·계절이 다른 사진 중
     * 한 장만 닮아도 같은 대상으로 인정한다. [refs] 가 비어 있거나 [captured] 가 null 이면 null.
     */
    fun maxCosineOrNull(captured: FloatArray?, refs: List<FloatArray>): Double? {
        if (captured == null || refs.isEmpty()) return null
        return refs.mapNotNull { cosineOrNull(captured, it) }.maxOrNull()
    }

    /**
     * 촬영본이 미션 문구([TextAnchors.positives])에 맞을 확률. CLIP 제로샷 분류와 같은 방식으로
     * `logitScale × 코사인` 을 softmax 에 넣되, 미션 문구 쪽은 가장 잘 맞는 하나(최대 logit)만 쓴다
     * (문구를 여러 개 넣었다고 확률이 부풀지 않도록). 대조 문구가 많을수록 우연히 높게 나올 확률이 준다.
     *
     * [captured] 가 null 이거나 [anchors] 가 비어 있거나 차원이 안 맞으면 null.
     */
    fun textMatchProbOrNull(captured: FloatArray?, anchors: PhotoVerification.TextAnchors?): Double? {
        if (captured == null || anchors == null || !anchors.isUsable) return null
        val pos = anchors.positives.mapNotNull { cosineOrNull(captured, it) }.maxOrNull() ?: return null
        val neg = anchors.negatives.mapNotNull { cosineOrNull(captured, it) }
        if (neg.isEmpty()) return null
        val logits = (listOf(pos) + neg).map { it * anchors.logitScale }
        val maxLogit = logits.max()
        val denom = logits.sumOf { exp(it - maxLogit) }
        return exp(logits[0] - maxLogit) / denom
    }

    /** L2 정규화한 복사본. 노름이 0이면 원본 복사본. */
    fun l2Normalized(v: FloatArray): FloatArray {
        var n = 0.0
        for (x in v) n += x.toDouble() * x
        if (n <= 0.0) return v.copyOf()
        val inv = (1.0 / sqrt(n)).toFloat()
        return FloatArray(v.size) { v[it] * inv }
    }
}

private fun FloatArray?.contentEqualsNullable(other: FloatArray?): Boolean = when {
    this == null && other == null -> true
    this == null || other == null -> false
    else -> this.contentEquals(other)
}
