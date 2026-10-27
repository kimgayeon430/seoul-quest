package smu.ai.graduation_project.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Test

class CategoryProbeTest {

    // 2차원 임베딩, 라벨 2개 + 무효. 투어 행은 x축, 맛집 행은 y축을 본다.
    private val model = CategoryProbe.Model(
        labels = listOf("투어", "맛집", PhotoVerification.INVALID_LABEL),
        weights = listOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(0.0, 2.0), doubleArrayOf(0.0, 0.0)),
        bias = doubleArrayOf(0.0, 0.0, 0.0),
        blend = 0.9,
    )
    private val scores = mapOf("투어" to 0.2, "맛집" to 0.5, PhotoVerification.INVALID_LABEL to 0.3)

    @Test
    fun probsAreSoftmaxOfLinearScores() {
        val p = CategoryProbe.probsOrNull(model, floatArrayOf(1f, 0f))!!
        val e2 = kotlin.math.exp(2.0)
        assertEquals(e2 / (e2 + 2), p.getValue("투어"), 1e-9)
        assertEquals(1 / (e2 + 2), p.getValue("맛집"), 1e-9)
    }

    @Test
    fun applyBlendsCategoriesButKeepsInvalidScore() {
        val c = CategoryProbe.apply(model, PhotoVerification.Classification(scores, floatArrayOf(1f, 0f)))
        val p = CategoryProbe.probsOrNull(model, floatArrayOf(1f, 0f))!!
        assertEquals(0.1 * 0.2 + 0.9 * p.getValue("투어"), c.scoreOf("투어"), 1e-9)
        assertEquals(0.1 * 0.5 + 0.9 * p.getValue("맛집"), c.scoreOf("맛집"), 1e-9)
        assertEquals(0.3, c.scoreOf(PhotoVerification.INVALID_LABEL), 1e-12)
        assertEquals("투어", c.topLabel)
    }

    @Test
    fun withoutEmbeddingOrModelClassificationIsUnchanged() {
        val noEmb = PhotoVerification.Classification(scores, null)
        assertSame(noEmb, CategoryProbe.apply(model, noEmb))
        val withEmb = PhotoVerification.Classification(scores, floatArrayOf(1f, 0f))
        assertSame(withEmb, CategoryProbe.apply(null, withEmb))
    }

    @Test
    fun dimensionMismatchIsIgnored() {
        assertNull(CategoryProbe.probsOrNull(model, floatArrayOf(1f, 0f, 0f)))
        val c = PhotoVerification.Classification(scores, floatArrayOf(1f, 0f, 0f))
        assertSame(c, CategoryProbe.apply(model, c))
    }

    @Test
    fun probeCanLiftWeakCategoryAboveHardReject() {
        // MobileViT 가 체험류를 약하게 보는 경우(0.10 < hardReject 0.22)라도 probe 가 확신하면 REJECT 를 면한다.
        val weak = mapOf("투어" to 0.10, "맛집" to 0.60, PhotoVerification.INVALID_LABEL to 0.30)
        val c = CategoryProbe.apply(model, PhotoVerification.Classification(weak, floatArrayOf(1f, 0f)))
        val r = PhotoVerification.verify("투어", c)
        assertEquals(PhotoVerification.Verdict.PASS, r.verdict)
    }
}
