package smu.ai.graduation_project.domain

/**
 * 사진 인증(2단계) 판정 임계값.
 *
 * 온디바이스 비전 모델의 오프라인 평가(test 셋 임계값 스윕)로 정한 값을 한곳에 모아,
 * 판정 결과를 설명하고 튜닝할 수 있게 한다. ([RecommendationWeights] 와 같은 역할)
 *
 * 현재 기본값: `mobilevit-small-fullft-1` (test macro-F1 0.81, 재촬영 스푸핑 augmentation 포함) 로 `ml/thresholds.json` 에서 선정.
 * invalidReject 0.55 → 무효 사진 차단율 ≈ 0.93 / 정상 사진 오탐율(hardReject 0.22 기준) ≈ 0.146.
 *
 * 유사도 임계값([similarityRescueThreshold] 0.50, [similaritySuspectThreshold] 0.68)은 미션
 * 대표 이미지 임베딩이 있을 때만 적용된다(보고서 6.7). 공개 scene 프록시(`ml/embedding_separability.py`)
 * 에서 같은 대상 쌍 p15 / 다른 대상 쌍 p90 으로 잡은 잠정치이며, 실제 미션 사진으로 재보정해야 한다.
 *
 * 텍스트 일치도 임계값([textMatchPassThreshold] 0.50, [textMatchSuspectThreshold] 0.15)은 미션 설명
 * 텍스트 임베딩이 있을 때만 적용된다. 값은 "미션 문구 1개 vs 대조 문구 N개(보통 15~25개)" softmax
 * 확률이라 우연 수준이 1/(N+1) ≈ 0.05 이다. 0.15 는 우연의 약 3배, 0.50 은 대조 문구 전부를 합친
 * 것보다 이 미션 문구에 더 가까운 수준이다. `ml/embed_mission_texts.py --eval` 의 대표 이미지 기준
 * 분포로 잡은 잠정치이며, `photoVerifyTextMatch` 실사용 로그로 재보정해야 한다.
 */
data class PhotoVerificationConfig(
    /** 미션 카테고리 점수가 이 값 이상이면 자동 통과. */
    val autoPassThreshold: Double = 0.65,
    /** 미션 카테고리 점수가 이 값 미만이면 재촬영을 요청한다. */
    val hardRejectThreshold: Double = 0.22,
    /** 무효(셀카·스크린샷·무관 실내 등) 점수가 이 값 이상이면 재촬영을 요청한다. */
    val invalidRejectThreshold: Double = 0.55,
    /**
     * 참조 이미지 유사도가 이 값 이상이면, 카테고리 점수가 낮아 `REJECT` 였을 사진을
     * 즉시 거절하지 않고 관리자 검수로 구제한다. (약한 카테고리 클래스 보완, 6.7.5-나)
     */
    val similarityRescueThreshold: Double = 0.50,
    /**
     * 참조 이미지 유사도의 "닮음" 기준선.
     *  - 이 값 이상 + 카테고리 애매 → 통과로 상향
     *  - 이 값 미만 + 카테고리 확실 → 같은 유형의 다른 대상 의심, 검수로 하향
     */
    val similaritySuspectThreshold: Double = 0.68,
    /**
     * 미션 설명 텍스트 일치 확률이 이 값 이상이면 "이 미션의 대상"이라는 강한 근거로 본다.
     *  - 카테고리 점수 낮음(REJECT) → 검수로 구제
     *  - 카테고리 애매(NEEDS_REVIEW) → 통과로 상향
     */
    val textMatchPassThreshold: Double = 0.50,
    /**
     * 미션 설명 텍스트 일치 확률이 이 값 미만이고 대표 이미지 유사도도 낮으면(또는 없으면),
     * 카테고리가 확실해도 "같은 유형의 다른 대상" 의심으로 검수로 하향한다.
     */
    val textMatchSuspectThreshold: Double = 0.15,
    /**
     * 모델을 불러오지 못했을 때(추론 불가) 미션을 어떻게 처리할지.
     * true 면 종전처럼 통과, false 면 관리자 검수로 보낸다.
     */
    val passWhenModelUnavailable: Boolean = false
) {
    init {
        require(hardRejectThreshold in 0.0..1.0) { "hardRejectThreshold 는 0..1 범위여야 합니다." }
        require(autoPassThreshold in 0.0..1.0) { "autoPassThreshold 는 0..1 범위여야 합니다." }
        require(invalidRejectThreshold in 0.0..1.0) { "invalidRejectThreshold 는 0..1 범위여야 합니다." }
        require(hardRejectThreshold <= autoPassThreshold) {
            "hardRejectThreshold 는 autoPassThreshold 보다 클 수 없습니다."
        }
        require(similarityRescueThreshold in -1.0..1.0) { "similarityRescueThreshold 는 -1..1 범위여야 합니다." }
        require(similaritySuspectThreshold in -1.0..1.0) { "similaritySuspectThreshold 는 -1..1 범위여야 합니다." }
        require(similarityRescueThreshold <= similaritySuspectThreshold) {
            "similarityRescueThreshold 는 similaritySuspectThreshold 보다 클 수 없습니다."
        }
        require(textMatchPassThreshold in 0.0..1.0) { "textMatchPassThreshold 는 0..1 범위여야 합니다." }
        require(textMatchSuspectThreshold in 0.0..1.0) { "textMatchSuspectThreshold 는 0..1 범위여야 합니다." }
        require(textMatchSuspectThreshold <= textMatchPassThreshold) {
            "textMatchSuspectThreshold 는 textMatchPassThreshold 보다 클 수 없습니다."
        }
    }

    companion object {
        val DEFAULT = PhotoVerificationConfig()
    }
}
