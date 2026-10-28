<div align="center">

# Seoul Quest

**미션을 따라 걷다 보면 서울 여행이 완성됩니다.**

온디바이스 다중 신호 사진 인증과 하이브리드 추천을 적용한 여행 미션 게이미피케이션 앱

![Android](https://img.shields.io/badge/Android-8.0%2B-3DDC84?logo=android&logoColor=white)
![Kotlin](https://img.shields.io/badge/Kotlin-2.2-7F52FF?logo=kotlin&logoColor=white)
![Jetpack Compose](https://img.shields.io/badge/Jetpack%20Compose-Material%203-4285F4?logo=jetpackcompose&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-Auth%20%C2%B7%20Firestore-FFCA28?logo=firebase&logoColor=black)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-on--device-005CED?logo=onnx&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-training-EE4C2C?logo=pytorch&logoColor=white)

</div>

**Seoul Quest**는 여행지 미션을 수행하고 GPS와 사진으로 인증해 포인트·레벨·배지를 얻는 Android 여행 앱입니다. 여행 취향에 맞는 미션 추천과 지도 탐색, 온디바이스 AI 사진 인증을 연결해 여행의 발견부터 성취까지 하나의 흐름으로 제공합니다.

숙명여자대학교 인공지능공학부 졸업 프로젝트로 개발했습니다.

### 한눈에 보기

- **온디바이스 4중 신호 사진 인증** — MobileViT 분류 + CLIP 선형 분류기 보정 + 대표 이미지 유사도 + 미션 문구 일치도를 결합해, 사진을 서버로 보내지 않고 기기에서 통과·검수·재촬영을 판정합니다. 같은 카테고리의 *다른 대상* 오수락을 75.5%에서 8.4%로 낮췄습니다.
- **하이브리드 미션 추천** — 6개 신호의 규칙 점수와 로지스틱 회귀 re-ranker를 결합하고 카테고리 다양성을 반영합니다.
- **사용자·관리자 양면 서비스** — 미션 수행·보상·랭킹·배지와, 관리자 미션 관리·사진 검수·제안 검수·사용자 관리를 한 앱에서 제공합니다.
- **재현 가능한 실험** — 데이터 구축, 학습, ONNX 변환, 평가 스크립트와 결과 JSON을 저장소에 모두 포함했습니다.

| 프로젝트 | 정보 |
| --- | --- |
| 팀 | SOOK-GO |
| 개발자 | 강규린 · 김가연 |
| 지도교수 | 김철연 |
| 개발 기간 | 2026.03–2026.10 |
| 플랫폼 | Android 8.0 이상 |
| 지원 언어 | 한국어 · English · 日本語 |

## 앱 다운로드

[**SeoulQuest.apk 내려받기 (v1.0)**](https://github.com/kimgayeon430/seoul-quest/releases/latest/download/SeoulQuest.apk) · [릴리스 페이지](https://github.com/kimgayeon430/seoul-quest/releases/latest)

- Android 8.0(API 26) 이상 기기에서 설치할 수 있으며, 파일 크기는 약 163MB입니다.
- 디버그 서명으로 배포하는 데모 APK입니다. 설치할 때 **출처를 알 수 없는 앱 설치**를 허용해야 하고, Play Protect 경고가 표시될 수 있습니다.
- 위치 인증과 사진 인증을 사용하려면 위치와 카메라 권한을 허용해야 합니다.

## 프로젝트 소개

Seoul Quest는 장소를 둘러보는 여행에 구체적인 활동과 달성 목표를 더합니다. 사용자는 투어·맛집·체험·쇼핑 중 관심 분야를 선택하고, 추천 미션이나 지도에서 다음 목적지를 찾습니다. 현장 방문과 사진 인증을 완료하면 보상을 받으며, 여행 레벨과 배지를 통해 활동을 확인할 수 있습니다.

사진 인증은 기기 내 AI 모델이 먼저 판정하고, 자동 판정이 어려운 경우 관리자 검수로 연결합니다. 사용자가 직접 미션을 제안하고 관리자가 검수하는 기능도 제공해 여행 콘텐츠를 함께 확장할 수 있습니다.

## 서비스 화면

실제 앱 실행 화면입니다. 미션 탐색, 완료 보상, 다음 미션 추천과 일본어 UI를 보여줍니다.

<table>
  <tr>
    <th>홈 · 맞춤 미션 추천</th>
    <th>카테고리별 미션 탐색</th>
    <th>지도에서 여행지 확인</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/home.jpg" alt="Seoul Quest 홈 화면과 추천 미션" width="220"></td>
    <td align="center"><img src="docs/missions.jpg" alt="카테고리별 미션 목록" width="220"></td>
    <td align="center"><img src="docs/map.jpg" alt="네이버 지도 위에 표시된 여행 미션" width="220"></td>
  </tr>
  <tr>
    <th>미션 완료 · 포인트 보상</th>
    <th>다음 미션 추천</th>
    <th>다국어 지원 · 일본어 화면</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/completed.jpg" alt="미션 완료 후 포인트와 레벨 진행률 표시" width="220"></td>
    <td align="center"><img src="docs/next-mission.jpg" alt="미션 완료 후 다음 추천 미션 카드" width="220"></td>
    <td align="center"><img src="docs/language.jpg" alt="일본어로 전환한 Seoul Quest 홈 화면" width="220"></td>
  </tr>
</table>

## 주요 기능

| 기능 | 설명 |
| --- | --- |
| 개인화 추천 | 여행 취향·완료 이력·거리·난이도·인기도·시간대를 반영한 미션 추천 |
| 미션 탐색 | 카테고리별 목록과 네이버 지도 탐색, 미션 상세·길찾기·진행 상태 확인 |
| GPS·사진 인증 | 목표 장소 반경 200m 이내 위치 인증과 온디바이스 AI 사진 판정 |
| 여행 보상 | 단계별 포인트, 사용자 랭킹, 여행 레벨 5단계와 배지 3종 |
| 다음 미션 추천 | 완료한 미션을 제외하고 다음에 수행할 미션 안내 |
| 참여형 콘텐츠 | 미션 제안·수정·검수 상태 확인, 좋아요와 찜한 미션 관리 |
| 다국어 지원 | 한국어·영어·일본어 UI 전환과 언어별 미션 콘텐츠 표시 |
| 계정 관리 | 회원가입·로그인, 여행 취향 설정, 게스트 둘러보기와 프로필 |

### 관리자 기능

`admins/{uid}` 문서가 있는 계정에는 하단 탭에 관리자 메뉴가 나타납니다. 관리자 권한은 Firestore 보안 규칙에서도 같은 기준으로 검사합니다.

| 화면 | 기능 |
| --- | --- |
| 미션 관리 | 미션 등록·수정·삭제, 카테고리·설명·위도/경도·포인트·소요 시간·대표 이미지 설정 |
| 사진 인증 검수 | AI가 `NEEDS_REVIEW`로 보낸 인증 사진을 예측 라벨·카테고리 점수·유사도·문구 일치도와 함께 확인하고 승인·반려 (승인 시 보상 지급) |
| 미션 제안 검수 | 사용자가 제안하거나 수정 요청한 미션을 승인·반려 |
| 사용자 관리 | 사용자별 진행 현황·포인트 조회, 관리자 권한 부여·해제 |

## 서비스 이용 흐름

1. **취향 설정** — 회원가입 후 관심 있는 여행 카테고리를 선택합니다.
2. **미션 선택** — 추천 목록이나 지도에서 미션과 목적지를 확인합니다.
3. **현장 방문** — 목표 지점 반경 200m 이내에서 GPS 위치 인증을 진행합니다.
4. **사진 인증** — 카메라로 촬영한 사진을 확인하고 AI 판정을 받습니다.
5. **보상 확인** — 인증이 완료되면 포인트·레벨·배지를 확인하고 다음 미션을 선택합니다.

### 사진 인증과 완료 처리

| 판정 | 처리 |
| --- | --- |
| 통과 (`PASS`) | 사진 업로드와 데이터 저장 성공 후 미션 완료 및 2단계 보상 지급 |
| 검수 대기 (`NEEDS_REVIEW`) | 사진을 저장하고 관리자 검수 대기. 승인 후 미션 완료 및 2단계 보상 지급 |
| 반려 (`REJECT`) | 사진을 업로드하지 않고 재촬영 안내 |

1단계 위치 인증 보상은 `min(미션 포인트, 100)`, 2단계 보상은 나머지 포인트입니다. 완료·보상 상태를 트랜잭션으로 갱신하고, 재시도와 중복 승인에 따른 중복 지급을 방지합니다.

### 여행 레벨과 배지

| 레벨 | 이름 | 누적 포인트 |
| --- | --- | ---: |
| Lv.1 | 여행 새싹 | 0P |
| Lv.2 | 동네 탐험가 | 500P |
| Lv.3 | 도시 여행자 | 1,500P |
| Lv.4 | 숨은 명소 수집가 | 3,000P |
| Lv.5 | 마스터 트래블러 | 5,000P |

배지는 첫 미션 완료 시 **첫 발자국**, 한 주에 미션 5개 완료 시 **주간 탐험가**, 같은 카테고리의 미션 3개 완료 시 **취향 발견**을 획득합니다. 프로필에서 다음 레벨까지의 진행률과 배지 획득 조건을 확인할 수 있습니다.

## 시스템 구성

![Seoul Quest 시스템 아키텍처 — Android 앱, 인증·데이터 저장, 지도 및 온디바이스 AI 구성](docs/system-architecture-final.png)

Android 앱은 Firebase로 계정과 미션 데이터를 관리하고, Supabase Storage에 인증 사진을 저장합니다. 사진 분류와 참조 이미지 유사도 계산은 ONNX Runtime으로 기기 내에서 수행합니다.

| 구성 요소 | 역할 |
| --- | --- |
| Android · Jetpack Compose | 사용자·관리자 화면, 내비게이션, 카메라와 위치 인증 |
| ViewModel · Repository · Domain | 미션 수행 상태 관리, 외부 서비스 연동, 인증·보상·추천 규칙 분리 |
| Firebase Authentication | 이메일·비밀번호 기반 사용자 인증 |
| Cloud Firestore | 사용자, 미션, 수행 이력, 검수 상태, 포인트 및 반응 데이터 저장 |
| Supabase Storage | 인증 사진·미션 대표 이미지 저장 |
| 네이버 지도 SDK | 미션 위치 마커, 진행 상태와 인증 사진 표시 |
| ONNX Runtime | MobileViT 사진 분류 및 CLIP 이미지 임베딩 추론 |
| Python 학습 파이프라인 | 사진 인증 모델 학습·평가·ONNX 변환, 추천 모델 학습·평가 |

ViewModel·Repository·Domain 분리는 미션 수행 기능에 적용했습니다. 인증·보상·추천의 핵심 규칙은 Firebase에 의존하지 않는 순수 Kotlin으로 구현합니다.

세부 구성도는 [`docs/diagrams/`](docs/diagrams)에 Mermaid 원본과 SVG·PNG로 있습니다: [시스템 구성](docs/diagrams/01-system-architecture.svg) · [서비스 흐름](docs/diagrams/02-service-flow.svg) · [사진 인증](docs/diagrams/03-photo-verification.svg) · [오프라인 ML 파이프라인](docs/diagrams/04-ml-offline-pipeline.svg) · [추천](docs/diagrams/05-recommender.svg) · [미션 제안](docs/diagrams/06-mission-proposal.svg)

### 기술 선택 이유

| 선택 | 이유 |
| --- | --- |
| Jetpack Compose | 사용자·관리자 화면을 선언형 UI와 공통 컴포넌트로 일관되게 구성 |
| Firebase Auth · Firestore | 별도 서버 없이 인증·실시간 데이터·보안 규칙을 제공하고, 트랜잭션으로 보상 중복 지급을 막을 수 있음 |
| Supabase Storage | 무료 범위에서 공개 이미지 URL을 바로 제공해 인증 사진·대표 이미지 저장에 적합 |
| 네이버 지도 SDK | 국내 지도 데이터와 한국어 지명 정확도 |
| 온디바이스 추론 (ONNX Runtime) | 사진을 서버로 보내기 전에 판정해 개인정보 노출과 서버 비용을 줄이고, 오프라인에서도 동작 |
| MobileViT-small | 모바일용 경량 ViT(약 5.6M 파라미터)로 정확도와 크기의 균형. 앱에 20MB ONNX로 포함 |
| CLIP ViT-B/32 | 한 임베딩으로 대표 이미지 비교·미션 문구 일치도·선형 분류기 보정을 모두 처리. int8 양자화 후 첫 사용 시 다운로드 |
| 로지스틱 회귀 re-ranker | 데이터가 적은 상황에서 과적합이 적고, 계수를 JSON으로 내보내 순수 Kotlin으로 추론 가능 |

## AI 설계 및 평가

### 모델 학습과 배포

| 단계 | 사진 인증 모델 | 추천 모델 |
| --- | --- | --- |
| 데이터 구축 | 투어·맛집·체험·쇼핑·무효의 5개 클래스, 총 6,063장 | 합성 사용자·미션 완료 로그 |
| 모델 학습 | MobileViT-small 파인튜닝 | 로지스틱 회귀 기반 re-ranker 학습 |
| 변환·저장 | ONNX 변환 (CLIP은 int8 양자화, 선형 분류기는 JSON) | 모델 계수를 JSON으로 저장 |
| 앱 적용 | ONNX Runtime으로 기기 내 사진 분석 | 규칙 점수와 모델 예측을 결합해 추천 |

### 온디바이스 사진 인증

**사진 촬영 → 무효 사진 차단 → 카테고리 분류 → 미션 대상 일치 확인 → 최종 판정**

![사진 인증 판정 흐름](docs/diagrams/03-photo-verification.png)

| 신호 | 모델 | 역할 |
| --- | --- | --- |
| 무효 점수 | MobileViT-small (파인튜닝) | 셀카·스크린샷·화면 재촬영 등 무효 사진을 가장 먼저 차단 |
| 카테고리 점수 | MobileViT + CLIP 선형 분류기 | 투어·맛집·체험·쇼핑 중 미션 카테고리와 맞는지 판단 |
| 대표 이미지 유사도 | CLIP 이미지 임베딩 | 관리자가 등록한 미션 대표 이미지와 촬영본의 코사인 유사도 |
| 미션 문구 일치도 | CLIP 텍스트 임베딩 | 미션의 시각 묘사 문구(`photoPromptsEn`)가 다른 미션 문구들보다 촬영본에 잘 맞는지 |

무효 게이트를 보조 신호보다 먼저 두어 스푸핑 사진이 유사도로 구제되지 않게 했습니다. 카테고리가 애매해도 대상이 확실히 일치하면 통과로 올리고, 카테고리가 확실해도 대상이 다르다고 의심되면 관리자 검수로 내립니다. 임계값은 [`PhotoVerificationConfig`](app/src/main/java/smu/ai/graduation_project/domain/PhotoVerificationConfig.kt)에 근거와 함께 모아 두었고, 판정별 저장·보상 처리는 [사진 인증과 완료 처리](#사진-인증과-완료-처리)를 참고하세요.

**카테고리 분류** (test 893장)

| 모델 | Macro-F1 [95% CI] | 무효 Recall | 정상 오거절 |
| --- | ---: | ---: | ---: |
| CLIP ViT-B/32 zero-shot | 0.627 [0.595, 0.660] | 0.667 | 0.327 |
| MobileViT-small 파인튜닝 | 0.811 [0.784, 0.838] | 0.932 | 0.146 |
| **MobileViT + CLIP 선형 분류기 (앱)** | **0.828** [0.801, 0.853] | **0.932** | **0.108** |

정상 사진 오거절 감소는 통계적으로 유의했습니다(McNemar p = 0.006).

**판정 파이프라인 구성 비교** (가상 미션 51개, 정상 533 · 같은 카테고리 다른 대상 510 · 무효 309건)

| 구성 | 정상 PASS ↑ | 정상 오거절 ↓ | 다른 대상 오수락 ↓ | 무효 오수락 ↓ |
| --- | ---: | ---: | ---: | ---: |
| 분류만 | 0.711 | 0.107 | 0.755 | 0.006 |
| + 대표 이미지 유사도 | 0.246 | 0.023 | 0.016 | 0.000 |
| + 미션 문구 일치도 | 0.822 | 0.036 | 0.076 | 0.000 |
| **전체 결합 (앱)** | **0.822** | **0.013** | **0.084** | **0.000** |

**테스트 기기 추론 속도** (2017년 출시, Android 9 · 1080×1440 사진, 중앙값)

| 단계 | 시간 |
| --- | ---: |
| MobileViT 분류 | 223 ms |
| CLIP 임베딩 | 214 ms |
| 전체 판정 (`PhotoGate.decide`) | 402 ms |

데이터셋은 공개 장면 데이터와 합성 무효 이미지로 구성한 6,063장이며, 수치는 실제 여행 현장 사진의 성능을 보장하지 않습니다. 대표 이미지·문구 임계값은 표본이 작은 잠정값으로, 실사용 로그로 재보정이 필요합니다. 전체 실험과 재현 방법은 [논문 원고](docs/report.md)와 [`ml/eval`](ml/eval/README.md)에 정리했습니다.

### 개인화 미션 추천

명시적 취향, 카테고리별 완료 이력, 난이도 적합도, 거리, 인기도, 시간대의 6개 신호를 사용합니다. 홈 화면은 규칙 기반 점수와 로지스틱 회귀 모델의 완료 확률을 결합하고, 카테고리 다양성을 반영해 상위 3개 미션을 선정합니다. 추천 이유도 함께 표시합니다.

| 지표 | 규칙 기반 | 학습 모델 | 하이브리드 · λ=0.6 |
| --- | ---: | ---: | ---: |
| Precision@3 | 0.914 | 0.951 | 0.947 |
| NDCG@5 | 0.914 | 0.953 | 0.947 |
| MAP | 0.873 | 0.904 | 0.899 |

추천 모델은 **합성 사용자·완료 로그**로 학습하고 사용자 단위로 분리한 테스트셋에서 평가했습니다. 실제 서비스 이용자의 추천 만족도나 성과를 측정한 결과는 아닙니다. 미션 완료 화면의 다음 추천은 규칙 기반 점수를 사용합니다.

## 기술 스택

| 영역 | 기술 |
| --- | --- |
| 앱 | Kotlin 2.2.10, Jetpack Compose, Material 3, Navigation Compose |
| 데이터·인증 | Firebase Authentication, Cloud Firestore, Supabase Storage |
| 지도·이미지 | 네이버 지도 SDK 3.23.3, Coil 2.7.0 |
| AI 학습 | Python, PyTorch, Hugging Face, scikit-learn |
| AI 추론 | ONNX Runtime Android 1.29.0, MobileViT-small, CLIP ViT-B/32 |
| 빌드 | Gradle 9.4.1, Android Gradle Plugin 9.2.0, Android SDK 36.1 |
| 검증 | JUnit4, Firebase Emulator, Firestore 규칙 테스트 |

## 프로젝트 구조

```text
app/src/main/
├── java/smu/ai/graduation_project/
│   ├── MainActivity.kt     # 앱 진입과 내비게이션
│   ├── data/               # Repository·외부 서비스·AI 추론
│   ├── domain/             # 인증·보상·추천 규칙
│   ├── model/              # 데이터 모델
│   ├── navigation/         # 화면 경로
│   └── ui/                 # 사용자·관리자 화면과 테마
├── assets/                 # ONNX 사진 분류 모델·CLIP 선형 분류기·추천 모델
└── res/                    # 한국어·영어·일본어 리소스
app/src/test/               # Kotlin 단위 테스트
app/src/androidTest/        # 실기기 ONNX 추론·속도 측정 테스트
firestore-tests/            # Firestore 보안 규칙 테스트
ml/                         # 사진 모델 데이터 구축·학습·ONNX 변환
├── eval/                   # 논문 실험 재현 스크립트와 결과 JSON
└── reco/                   # 추천 모델 학습·평가
docs/                       # 논문 원고, 구성도, 실행 화면, 포스터
firestore.rules             # 데이터 접근 규칙
```

## 실행 방법

### 준비 환경

- Android Studio, JDK 17 이상, Android SDK 36.1
- Android 8.0(API 26) 이상 기기 또는 에뮬레이터 · `arm64-v8a` / `x86_64`
- Firebase 프로젝트 · Email/Password 인증 및 Cloud Firestore 활성화
- Supabase 프로젝트 · 이미지 저장용 Storage
- 네이버 클라우드 플랫폼 Maps 애플리케이션 및 Client ID

### 프로젝트 설정

```bash
git clone https://github.com/kimgayeon430/seoul-quest.git
cd seoul-quest
```

1. Android Studio에서 프로젝트를 엽니다.
2. 저장소의 `app/google-services.json`은 개발에 사용한 Firebase 프로젝트 설정입니다. 다른 Firebase 프로젝트를 쓰려면 패키지 `smu.ai.graduation_project`를 등록하고 발급받은 파일로 교체합니다.
3. Supabase Storage에 공개 버킷 `mission-photos`를 생성하고 앱에서 사용할 업로드 정책을 설정합니다. 프로젝트에서 사용하는 기본 INSERT 정책은 다음과 같습니다.

   ```sql
   create policy "anon upload mission-photos"
   on storage.objects for insert to anon
   with check (bucket_id = 'mission-photos');
   ```

4. 네이버 Maps 애플리케이션에 Android 패키지를 등록합니다.
5. 프로젝트 루트의 `local.properties`에 다음 값을 설정합니다. Android Studio가 작성한 `sdk.dir`은 유지합니다.

   ```properties
   SUPABASE_URL=https://<project>.supabase.co
   SUPABASE_ANON_KEY=<anon-or-publishable-key>
   NAVER_MAP_CLIENT_ID=<maps-client-id>
   ```

6. 저장소의 Firestore 규칙을 대상 Firebase 프로젝트에 배포합니다.

   ```bash
   firebase login
   firebase use <firebase-project-id>
   firebase deploy --only firestore:rules
   ```

   규칙 배포에는 Firebase CLI가 필요합니다.

7. Gradle Sync 후 연결된 기기에서 앱을 실행합니다.

   ```bash
   ./gradlew installDebug
   ```

`local.properties`의 키는 저장소에 포함하지 않습니다. 값이 없어도 빌드와 테스트는 되지만, 지도 표시와 사진 업로드는 동작하지 않습니다. 위치 인증과 촬영 시 위치·카메라 권한을 허용해야 합니다. 사진 분류 모델을 불러오지 못하면 관리자 검수로 연결하며, CLIP 모델은 첫 사용 시 Hugging Face Hub에서 다운로드 후 기기에 캐시합니다.

### 테스트

```bash
./gradlew :app:testDebugUnitTest                  # 도메인 규칙 단위 테스트 (129개)
cd firestore-tests && npm install && npm test     # Firestore 보안 규칙 테스트 (Emulator)
```

실기기 추론 속도·메모리 측정은 [측정 가이드](docs/benchmark-guide.md)를 참고하세요.

도메인 규칙 단위 테스트와 Firestore Emulator 테스트로 인증·보상·접근 권한을 검증합니다. 사용자 미션 제안·검수·좋아요·찜 기능은 빌드와 자동 테스트 기준으로 확인했으며, 전체 사용자 흐름의 실기기 검증 범위와는 구분합니다.

## 적용 범위와 확장 방향

졸업 프로젝트의 구현 범위는 여행 미션 탐색·인증·보상, 개인화 추천, 다국어 UI와 관리자 검수입니다. 서비스 운영 단계에서는 실제 여행 사진과 이용 로그를 통한 모델 보정, 서버 측 포인트·보상 검증, 위치 조작 탐지와 사용자 흐름 테스트 확대가 필요합니다.

## 관련 문서

- [졸업 논문 원고](docs/report.md) · [Word 버전](docs/report.docx) · [프로젝트 포스터](docs/project-poster.pdf)
- [개발 기록](docs/dev-notes.md)
- [실험 재현 방법](ml/eval/README.md)
- [실기기 속도 측정 가이드](docs/benchmark-guide.md)
- [사진 인증 모델과 학습 파이프라인](ml/README.md)
- [추천 모델과 평가 방법](ml/reco/README.md)
- [사진 인증 평가 설정](ml/thresholds.json)

## 팀

| 이름 | 소속 |
| --- | --- |
| 강규린 | 숙명여자대학교 인공지능공학부 |
| 김가연 | 숙명여자대학교 인공지능공학부 |

지도교수 김철연 · 팀 SOOK-GO
