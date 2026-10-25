package smu.ai.graduation_project.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** 사진 인증(2단계) 판정 규칙. */
class PhotoVerificationTest {

    private fun classification(vararg pairs: Pair<String, Double>) =
        PhotoVerification.Classification(pairs.toMap())

    private fun classification(embedding: FloatArray, vararg pairs: Pair<String, Double>) =
        PhotoVerification.Classification(pairs.toMap(), embedding)

    @Test
    fun highCategoryScorePasses() {
        val result = PhotoVerification.verify(
            missionCategory = "맛집",
            classification = classification("맛집" to 0.90, PhotoVerification.INVALID_LABEL to 0.02)
        )
        assertEquals(PhotoVerification.Verdict.PASS, result.verdict)
    }

    @Test
    fun highInvalidScoreIsRejected() {
        val result = PhotoVerification.verify(
            missionCategory = "맛집",
            classification = classification("맛집" to 0.20, PhotoVerification.INVALID_LABEL to 0.75)
        )
        assertEquals(PhotoVerification.Verdict.REJECT, result.verdict)
    }

    @Test
    fun lowCategoryScoreIsRejected() {
        val result = PhotoVerification.verify(
            missionCategory = "체험",
            classification = classification(
                "체험" to 0.10, "투어" to 0.50, PhotoVerification.INVALID_LABEL to 0.10
            )
        )
        assertEquals(PhotoVerification.Verdict.REJECT, result.verdict)
    }

    @Test
    fun midCategoryScoreNeedsReview() {
        val result = PhotoVerification.verify(
            missionCategory = "쇼핑",
            classification = classification("쇼핑" to 0.50, PhotoVerification.INVALID_LABEL to 0.10)
        )
        assertEquals(PhotoVerification.Verdict.NEEDS_REVIEW, result.verdict)
    }

    @Test
    fun nullClassificationNeedsReviewByDefault() {
        val result = PhotoVerification.verify(missionCategory = "투어", classification = null)
        assertEquals(PhotoVerification.Verdict.NEEDS_REVIEW, result.verdict)
    }

    @Test
    fun nullClassificationPassesWhenConfigured() {
        val result = PhotoVerification.verify(
            missionCategory = "투어",
            classification = null,
            config = PhotoVerificationConfig(passWhenModelUnavailable = true)
        )
        assertEquals(PhotoVerification.Verdict.PASS, result.verdict)
    }

    @Test
    fun boundaryScoreEqualToAutoPassThresholdPasses() {
        val config = PhotoVerificationConfig(autoPassThreshold = 0.70)
        val result = PhotoVerification.verify(
            missionCategory = "투어",
            classification = classification("투어" to 0.70, PhotoVerification.INVALID_LABEL to 0.0),
            config = config
        )
        assertEquals(PhotoVerification.Verdict.PASS, result.verdict)
    }

    // ---- 참조 이미지 유사도 결합 (보고서 6.7.5) ----

    /** cos 유사도가 목표가 되도록 방향을 맞춘 2D 벡터. `ref` 와의 각도로 유사도를 만든다. */
    private val ref = floatArrayOf(1f, 0f)
    private val simHigh = floatArrayOf(1f, 0f)          // cos = 1.0
    private val simMid = floatArrayOf(0.6f, 0.8f)       // cos = 0.6  (rescue..suspect 사이)
    private val simLow = floatArrayOf(0f, 1f)           // cos = 0.0

    @Test
    fun highCategoryHighSimilarityPasses() {
        val r = PhotoVerification.verify(
            "투어", classification(simHigh, "투어" to 0.90, PhotoVerification.INVALID_LABEL to 0.02),
            referenceEmbeddings = listOf(ref)
        )
        assertEquals(PhotoVerification.Verdict.PASS, r.verdict)
        assertEquals(1.0, r.similarity!!, 1e-6)
    }

    @Test
    fun highCategoryLowSimilarityIsDowngradedToReview() {
        // 유형은 확실한데 대표 이미지와 안 닮음 → 같은 유형의 다른 대상 의심.
        val r = PhotoVerification.verify(
            "투어", classification(simLow, "투어" to 0.90, PhotoVerification.INVALID_LABEL to 0.02),
            referenceEmbeddings = listOf(ref)
        )
        assertEquals(PhotoVerification.Verdict.NEEDS_REVIEW, r.verdict)
    }

    @Test
    fun lowCategoryHighSimilarityIsRescuedToReview() {
        // 카테고리 점수는 낮지만 대표 이미지와 닮아 즉시 거절하지 않는다 (약한 클래스 구제).
        val r = PhotoVerification.verify(
            "체험", classification(simHigh, "체험" to 0.10, "투어" to 0.5, PhotoVerification.INVALID_LABEL to 0.05),
            referenceEmbeddings = listOf(ref)
        )
        assertEquals(PhotoVerification.Verdict.NEEDS_REVIEW, r.verdict)
    }

    @Test
    fun midCategoryHighSimilarityIsUpgradedToPass() {
        val r = PhotoVerification.verify(
            "쇼핑", classification(simHigh, "쇼핑" to 0.50, PhotoVerification.INVALID_LABEL to 0.10),
            referenceEmbeddings = listOf(ref)
        )
        assertEquals(PhotoVerification.Verdict.PASS, r.verdict)
    }

    @Test
    fun midCategoryMidSimilarityStaysReview() {
        val r = PhotoVerification.verify(
            "쇼핑", classification(simMid, "쇼핑" to 0.50, PhotoVerification.INVALID_LABEL to 0.10),
            referenceEmbeddings = listOf(ref)
        )
        assertEquals(PhotoVerification.Verdict.NEEDS_REVIEW, r.verdict)
    }

    @Test
    fun lowCategoryLowSimilarityIsRejected() {
        val r = PhotoVerification.verify(
            "체험", classification(simLow, "체험" to 0.10, "투어" to 0.5, PhotoVerification.INVALID_LABEL to 0.05),
            referenceEmbeddings = listOf(ref)
        )
        assertEquals(PhotoVerification.Verdict.REJECT, r.verdict)
    }

    @Test
    fun invalidRejectShortCircuitsBeforeSimilarity() {
        // 참조 이미지를 화면에 띄워 재촬영하면 유사도는 최대에 가깝다. 무효가 먼저 잘라야 한다.
        val r = PhotoVerification.verify(
            "맛집", classification(simHigh, "맛집" to 0.20, PhotoVerification.INVALID_LABEL to 0.80),
            referenceEmbeddings = listOf(ref)
        )
        assertEquals(PhotoVerification.Verdict.REJECT, r.verdict)
        assertNull("무효 단락 시 유사도는 계산하지 않는다", r.similarity)
    }

    @Test
    fun noReferenceEmbeddingKeepsCategoryOnlyVerdict() {
        val r = PhotoVerification.verify(
            "체험", classification(simHigh, "체험" to 0.10, PhotoVerification.INVALID_LABEL to 0.05),
            referenceEmbeddings = emptyList()
        )
        assertEquals(PhotoVerification.Verdict.REJECT, r.verdict)
        assertNull(r.similarity)
    }

    @Test
    fun missingCapturedEmbeddingSkipsSimilarity() {
        val r = PhotoVerification.verify(
            "체험", classification("체험" to 0.10, PhotoVerification.INVALID_LABEL to 0.05),
            referenceEmbeddings = listOf(ref)
        )
        assertEquals(PhotoVerification.Verdict.REJECT, r.verdict)
        assertNull(r.similarity)
    }

    @Test
    fun multipleReferencesUseMaxSimilarity() {
        // 참조 이미지 두 장 중 하나만 닮아도(각도·조명이 다른 사진) 최대 유사도로 구제된다(6.7.8).
        val r = PhotoVerification.verify(
            "쇼핑", classification(simHigh, "쇼핑" to 0.50, PhotoVerification.INVALID_LABEL to 0.10),
            referenceEmbeddings = listOf(simLow, ref)
        )
        assertEquals(PhotoVerification.Verdict.PASS, r.verdict)
        assertEquals(1.0, r.similarity!!, 1e-6)
    }

    @Test
    fun maxCosineOrNullPicksBestReference() {
        assertEquals(
            1.0,
            PhotoEmbedding.maxCosineOrNull(floatArrayOf(1f, 0f), listOf(floatArrayOf(0f, 1f), floatArrayOf(1f, 0f)))!!,
            1e-9
        )
    }

    @Test
    fun maxCosineOrNullNullWhenNoReferencesOrNoCapture() {
        assertNull(PhotoEmbedding.maxCosineOrNull(floatArrayOf(1f, 0f), emptyList()))
        assertNull(PhotoEmbedding.maxCosineOrNull(null, listOf(ref)))
    }

    @Test
    fun cosineHandlesNullAndMismatchedLengths() {
        assertNull(PhotoEmbedding.cosineOrNull(null, ref))
        assertNull(PhotoEmbedding.cosineOrNull(floatArrayOf(1f, 2f, 3f), floatArrayOf(1f, 2f)))
        assertEquals(1.0, PhotoEmbedding.cosineOrNull(floatArrayOf(2f, 0f), floatArrayOf(5f, 0f))!!, 1e-9)
        assertEquals(0.0, PhotoEmbedding.cosineOrNull(floatArrayOf(0f, 1f), floatArrayOf(1f, 0f))!!, 1e-9)
    }

    @Test
    fun l2NormalizedHasUnitNorm() {
        val n = PhotoEmbedding.l2Normalized(floatArrayOf(3f, 4f))
        val norm = kotlin.math.sqrt((n[0] * n[0] + n[1] * n[1]).toDouble())
        assertTrue(kotlin.math.abs(norm - 1.0) < 1e-6)
    }

    // ---- 미션 설명 텍스트 일치도 ----

    // 2차원 단위 벡터로 텍스트 앵커를 만든다. 촬영본 (1,0) 기준 미션 문구 쪽이 가까우면 확률이 높다.
    private val photo = floatArrayOf(1f, 0f)
    private fun anchors(posCos: Float, negCos: Float, negCount: Int = 10) = PhotoVerification.TextAnchors(
        positives = listOf(unit(posCos)),
        negatives = List(negCount) { unit(negCos) },
    )
    private fun unit(cos: Float) = floatArrayOf(cos, kotlin.math.sqrt(1f - cos * cos))

    @Test
    fun textMatchProbMatchesSoftmax() {
        // logit 차이 = 100 × (0.30 - 0.25) = 5 → p = e^5 / (e^5 + 10)
        val p = PhotoEmbedding.textMatchProbOrNull(photo, anchors(0.30f, 0.25f))!!
        val e5 = kotlin.math.exp(5.0)
        assertEquals(e5 / (e5 + 10), p, 1e-4)
    }

    @Test
    fun textMatchProbUsesBestPositiveOnly() {
        // 미션 문구를 여러 개 넣어도 확률이 부풀지 않는다(최대 logit 하나만 사용).
        val one = PhotoEmbedding.textMatchProbOrNull(photo, anchors(0.30f, 0.25f))!!
        val many = PhotoEmbedding.textMatchProbOrNull(
            photo,
            PhotoVerification.TextAnchors(List(5) { unit(0.30f) }, List(10) { unit(0.25f) })
        )!!
        assertEquals(one, many, 1e-9)
    }

    @Test
    fun textMatchProbNullWithoutNegativesOrEmbedding() {
        assertNull(PhotoEmbedding.textMatchProbOrNull(photo, PhotoVerification.TextAnchors(listOf(photo), emptyList())))
        assertNull(PhotoEmbedding.textMatchProbOrNull(null, anchors(0.30f, 0.25f)))
        assertNull(PhotoEmbedding.textMatchProbOrNull(photo, null))
    }

    @Test
    fun highCategoryTextMismatchIsDowngradedToReview() {
        // 대표 이미지 없이 텍스트만 있는 미션: 맛집 사진은 확실하지만 이 미션 문구와 안 맞음 → 다른 식당 의심.
        val r = PhotoVerification.verify(
            "맛집", classification(photo, "맛집" to 0.90, PhotoVerification.INVALID_LABEL to 0.02),
            textAnchors = anchors(0.20f, 0.30f)
        )
        assertEquals(PhotoVerification.Verdict.NEEDS_REVIEW, r.verdict)
        assertTrue(r.textMatch!! < PhotoVerificationConfig.DEFAULT.textMatchSuspectThreshold)
        assertNull(r.similarity)
    }

    @Test
    fun highCategoryTextMatchPasses() {
        val r = PhotoVerification.verify(
            "맛집", classification(photo, "맛집" to 0.90, PhotoVerification.INVALID_LABEL to 0.02),
            textAnchors = anchors(0.32f, 0.25f)
        )
        assertEquals(PhotoVerification.Verdict.PASS, r.verdict)
    }

    @Test
    fun lowCategoryStrongTextMatchIsRescuedToReview() {
        // 약한 클래스(체험)라 카테고리 점수가 낮아도 미션 문구와 확실히 맞으면 즉시 거절하지 않는다.
        val r = PhotoVerification.verify(
            "체험", classification(photo, "체험" to 0.10, "투어" to 0.5, PhotoVerification.INVALID_LABEL to 0.05),
            textAnchors = anchors(0.32f, 0.25f)
        )
        assertEquals(PhotoVerification.Verdict.NEEDS_REVIEW, r.verdict)
    }

    @Test
    fun midCategoryStrongTextMatchIsUpgradedToPass() {
        val r = PhotoVerification.verify(
            "쇼핑", classification(photo, "쇼핑" to 0.50, PhotoVerification.INVALID_LABEL to 0.10),
            textAnchors = anchors(0.32f, 0.25f)
        )
        assertEquals(PhotoVerification.Verdict.PASS, r.verdict)
    }

    @Test
    fun textMatchKeepsPassWhenReferenceImageLooksDifferent() {
        // 대표 이미지와 각도가 달라 유사도는 낮지만 미션 문구와는 맞음 → 검수로 내리지 않는다.
        val r = PhotoVerification.verify(
            "투어", classification(photo, "투어" to 0.90, PhotoVerification.INVALID_LABEL to 0.02),
            referenceEmbeddings = listOf(floatArrayOf(0f, 1f)),
            textAnchors = anchors(0.30f, 0.27f)
        )
        assertEquals(PhotoVerification.Verdict.PASS, r.verdict)
        assertEquals(0.0, r.similarity!!, 1e-6)
    }

    @Test
    fun bothSignalsLowDowngradesToReview() {
        val r = PhotoVerification.verify(
            "투어", classification(photo, "투어" to 0.90, PhotoVerification.INVALID_LABEL to 0.02),
            referenceEmbeddings = listOf(floatArrayOf(0f, 1f)),
            textAnchors = anchors(0.20f, 0.30f)
        )
        assertEquals(PhotoVerification.Verdict.NEEDS_REVIEW, r.verdict)
    }

    @Test
    fun invalidRejectShortCircuitsBeforeTextMatch() {
        val r = PhotoVerification.verify(
            "맛집", classification(photo, "맛집" to 0.20, PhotoVerification.INVALID_LABEL to 0.80),
            textAnchors = anchors(0.35f, 0.20f)
        )
        assertEquals(PhotoVerification.Verdict.REJECT, r.verdict)
        assertNull(r.textMatch)
    }
}
