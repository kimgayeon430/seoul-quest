package smu.ai.graduation_project.data

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import smu.ai.graduation_project.domain.PhotoEmbedding
import smu.ai.graduation_project.domain.PhotoVerification
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.random.Random

/**
 * 논문용 온디바이스 비용 측정. 정확도가 아니라 **시간·메모리**만 잰다.
 *
 * 앱의 실제 코드 경로를 그대로 탄다: 분류는 [OnnxPhotoVerifier], 임베딩은 [OnnxClipPhotoEmbedder]
 * (+ [smu.ai.graduation_project.domain.CategoryProbe]), 전체 판정은 [PhotoGate.decide].
 *
 *     ./gradlew :app:connectedDebugAndroidTest \
 *       -Pandroid.testInstrumentationRunnerArguments.class=smu.ai.graduation_project.data.DeviceBenchmarkTest \
 *       -Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true   # 결과 파일을 adb pull 하려면
 *
 * 선택 인자(같은 방식으로 -P...runnerArguments.<이름>=<값>):
 *  - `network=wifi|lte|5g`  : 임베더 다운로드를 잴 때 연결 종류를 결과에 적는다(앱에 네트워크 권한이 없어 자동 판별 안 함)
 *  - `forceDownload=true`   : 캐시된 CLIP 임베더를 지우고 다시 받아 다운로드 시간을 잰다
 *  - `runs=20`, `warmup=3`  : 반복 횟수
 *
 * 결과: Logcat 태그 `DeviceBenchmark` 에 Markdown 표, 그리고
 * `/sdcard/Android/data/smu.ai.graduation_project/files/device_benchmark.md` 파일. 사용법은 `docs/benchmark-guide.md`.
 */
@RunWith(AndroidJUnit4::class)
class DeviceBenchmarkTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val args = InstrumentationRegistry.getArguments()

    private val runs = args.getString("runs")?.toIntOrNull() ?: 20
    private val warmup = args.getString("warmup")?.toIntOrNull() ?: 3

    private var peakJavaMb = 0.0
    private var peakNativeMb = 0.0

    private fun sampleMemory() {
        val rt = Runtime.getRuntime()
        peakJavaMb = maxOf(peakJavaMb, (rt.totalMemory() - rt.freeMemory()) / 1_048_576.0)
        peakNativeMb = maxOf(peakNativeMb, Debug.getNativeHeapAllocatedSize() / 1_048_576.0)
    }

    /** 카메라 촬영본과 비슷한 크기의 JPEG. 내용은 시간 측정과 무관하므로 그라데이션+잡음으로 만든다. */
    private fun photo(width: Int, height: Int, seed: Int): ByteArray {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawPaint(Paint().apply {
            shader = LinearGradient(0f, 0f, width.toFloat(), height.toFloat(),
                Color.rgb(90, 140, 200), Color.rgb(220, 170, 90), Shader.TileMode.CLAMP)
        })
        val rnd = Random(seed)
        val p = Paint()
        repeat(400) {
            p.color = Color.rgb(rnd.nextInt(256), rnd.nextInt(256), rnd.nextInt(256))
            canvas.drawCircle(rnd.nextFloat() * width, rnd.nextFloat() * height, rnd.nextFloat() * width / 12, p)
        }
        return ByteArrayOutputStream().use { out ->
            bmp.compress(Bitmap.CompressFormat.JPEG, 92, out)
            bmp.recycle()
            out.toByteArray()
        }
    }

    private fun unit(dim: Int, rnd: Random): FloatArray =
        PhotoEmbedding.l2Normalized(FloatArray(dim) { rnd.nextFloat() - 0.5f })

    private inline fun timeMs(block: () -> Unit): Double {
        val t0 = SystemClock.elapsedRealtimeNanos()
        block()
        return (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
    }

    private data class Stat(val name: String, val photo: String, val median: Double, val p90: Double, val n: Int)

    private fun bench(name: String, photo: String, block: () -> Unit): Stat {
        repeat(warmup) { block(); sampleMemory() }
        val xs = List(runs) { timeMs(block).also { sampleMemory() } }.sorted()
        val p90 = xs[((xs.size - 1) * 0.9).toInt()]
        return Stat(name, photo, xs[xs.size / 2], p90, xs.size)
    }

    @Test
    fun measure() {
        val lines = mutableListOf<String>()
        fun out(s: String = "") { lines += s }

        out("# 온디바이스 측정 결과")
        out()
        out("| 항목 | 값 |")
        out("| --- | --- |")
        out("| 기기 | ${Build.MANUFACTURER} ${Build.MODEL} |")
        if (Build.VERSION.SDK_INT >= 31) out("| SoC | ${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL} |")
        out("| Android | ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}) |")
        out("| 반복 | 워밍업 $warmup 회 + 측정 $runs 회 (중앙값, p90) |")

        // ---- 1) CLIP 임베더 다운로드 (캐시 없을 때만, 또는 forceDownload)
        val modelDir = File(context.filesDir, "models")
        if (args.getString("forceDownload") == "true") {
            modelDir.listFiles()?.filter { it.name.startsWith("photo_embedder") }?.forEach { it.delete() }
        }
        val source = PhotoEmbedderAssets.source(context)
        val cachedBefore = source.open()?.use { true } ?: false
        if (!cachedBefore) {
            val ms = timeMs { source.prefetch() }
            val size = File(modelDir, "photo_embedder_int8.onnx").length() / 1_048_576.0
            val net = args.getString("network") ?: "미기재"
            out("| CLIP 임베더 다운로드 | ${"%.1f".format(ms / 1000)} 초 (${"%.1f".format(size)} MB, 네트워크: $net) |")
        } else {
            out("| CLIP 임베더 다운로드 | 이미 캐시됨 (다시 재려면 forceDownload=true) |")
        }
        assertNotNull("CLIP 임베더를 받지 못했다 (인터넷 연결 확인)", source.open()?.use { true })

        // ---- 2) 모델 첫 로드 (세션 생성 포함 첫 호출)
        val sample = photo(1080, 1440, 1)
        val classifierOnly = OnnxPhotoVerifier(context, embedder = null, probeAsset = null)
        val embedder = OnnxClipPhotoEmbedder(context, source)
        val full = OnnxPhotoVerifier(context, embedder = embedder)  // 앱과 같은 구성: 분류기 + 임베더 + 선형 분류기
        sampleMemory()
        val firstClassify = timeMs { assertNotNull(classifierOnly.classify(sample)) }
        val firstEmbed = timeMs { assertNotNull(embedder.embed(sample)) }
        full.classify(sample)
        out("| 첫 호출 (모델 로드 포함) | 분류 ${"%.0f".format(firstClassify)} ms · 임베딩 ${"%.0f".format(firstEmbed)} ms |")
        out()

        // ---- 3) 반복 측정: 사진 크기 두 가지
        val rnd = Random(7)
        val anchors = PhotoVerification.TextAnchors(
            positives = List(2) { unit(512, rnd) },
            negatives = List(110) { unit(512, rnd) },   // 실제 미션과 같은 규모의 대조 문장
        )
        val refs = List(3) { unit(512, rnd) }
        val stats = mutableListOf<Stat>()
        for ((label, w, h) in listOf(Triple("1080×1440", 1080, 1440), Triple("3024×4032", 3024, 4032))) {
            val bytes = photo(w, h, w)
            stats += bench("① MobileViT 분류 (디코드·전처리 포함)", label) { classifierOnly.classify(bytes) }
            stats += bench("② CLIP 임베딩 (디코드·전처리 포함)", label) { embedder.embed(bytes) }
            stats += bench("③ 전체 판정 PhotoGate.decide", label) {
                PhotoGate.decide(bytes, "투어", full, referenceEmbeddings = refs, textAnchors = anchors)
            }
        }
        out("| 단계 | 사진 크기 | 중앙값 (ms) | p90 (ms) | n |")
        out("| --- | --- | ---: | ---: | ---: |")
        stats.forEach { out("| ${it.name} | ${it.photo} | ${"%.0f".format(it.median)} | ${"%.0f".format(it.p90)} | ${it.n} |") }
        out()
        out("③ 전체 판정 = 사진 디코드 → MobileViT 분류 → CLIP 임베딩 → 선형 분류기 결합 → 참조 이미지 유사도·문장 일치도 → 판정.")
        out("최대 메모리(측정 중 샘플링): Java 힙 ${"%.0f".format(peakJavaMb)} MB · 네이티브 힙 ${"%.0f".format(peakNativeMb)} MB")

        val report = lines.joinToString("\n")
        report.lines().forEach { Log.i(TAG, it) }
        val file = File(context.getExternalFilesDir(null), "device_benchmark.md")
        file.writeText(report + "\n")
        Log.i(TAG, "결과 파일: ${file.absolutePath}")
    }

    companion object {
        private const val TAG = "DeviceBenchmark"
    }
}
