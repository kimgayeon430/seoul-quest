# 실기기 속도 측정 가이드 (논문용)

휴대폰에서 사진 인증 AI가 얼마나 빨리 도는지 재는 방법입니다. 약 15분이면 끝납니다.
정확도는 재지 않고 **시간과 메모리만** 잽니다. 결과는 표 하나로 나옵니다.

## 1. 준비물

- 노트북: **Android Studio** (최신 버전이면 JDK 21과 Android SDK가 함께 설치됨), **Git**
- 휴대폰: Android 8.0 이상, 개발자 옵션에서 **USB 디버깅 켜기**, USB 케이블
  (설정 → 휴대전화 정보 → 빌드 번호 7번 탭 → 개발자 옵션 → USB 디버깅)
- 인터넷: 처음 한 번 AI 모델(약 89MB)을 받습니다. 가능하면 **Wi-Fi**에서 하세요.
- 비밀 키: 이 측정에는 **필요 없습니다.** `local.properties`의 `NAVER_MAP_CLIENT_ID`, `SUPABASE_URL`,
  `SUPABASE_ANON_KEY`가 없어도 빌드·측정이 됩니다. (앱을 직접 써 보려면 가연에게 값을 받으세요.)

> Windows 이고 사용자 폴더 이름에 한글이 있으면 Gradle 테스트가 실패할 수 있습니다.
> 시스템 환경 변수 `GRADLE_USER_HOME` 을 `C:\gradle-home` 처럼 한글 없는 경로로 설정한 뒤 터미널을 다시 여세요.

## 2. 코드 받기

```bash
git clone https://github.com/kimgayeon430/seoul-quest.git
cd seoul-quest
```

Android Studio 로 이 폴더를 한 번 열어 Gradle 동기화가 끝날 때까지 기다리면 `local.properties`(SDK 경로)가 자동으로 생깁니다.

## 3. 측정 (명령 한 줄)

휴대폰을 USB 로 연결하고, 휴대폰 화면의 "USB 디버깅 허용"을 누른 뒤:

```bash
# macOS / Linux
./gradlew :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=smu.ai.graduation_project.data.DeviceBenchmarkTest -Pandroid.testInstrumentationRunnerArguments.network=wifi -Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true

# Windows (PowerShell / cmd)
gradlew.bat :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=smu.ai.graduation_project.data.DeviceBenchmarkTest -Pandroid.testInstrumentationRunnerArguments.network=wifi -Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true
```

- `network=` 에는 연결 종류를 적습니다: `wifi` / `lte` / `5g`.
- `BUILD SUCCESSFUL` 이 나오면 끝입니다. 측정 중에는 휴대폰을 만지지 말고 화면을 켜 두세요.
- 모델 다운로드 시간을 다시 재고 싶으면 명령 끝에 `-Pandroid.testInstrumentationRunnerArguments.forceDownload=true` 를 붙입니다.

## 4. 결과 꺼내기

셋 중 편한 방법 하나면 됩니다.

1. **파일로 받기** (위 명령의 `leaveApksInstalledAfterRun=true` 덕분에 앱이 남아 있어야 함)
   ```bash
   adb pull /sdcard/Android/data/smu.ai.graduation_project/files/device_benchmark.md
   ```
2. **Logcat 에서 복사**
   ```bash
   adb logcat -d -s DeviceBenchmark:I
   ```
   Android Studio 아래쪽 Logcat 창에서 `tag:DeviceBenchmark` 로 검색해도 됩니다.
3. **빌드 폴더에서 찾기**: `app/build/outputs/androidTest-results/connected/` 아래 `logcat-...DeviceBenchmarkTest-measure.txt` 파일

(`adb` 는 Android SDK 의 `platform-tools` 폴더에 있습니다. Android Studio 의 Terminal 탭에서 실행하면 바로 됩니다.)

## 5. 보내 줄 것

- `device_benchmark.md` 파일 (또는 내용 복사/스크린샷)
- 휴대폰 이름과 측정할 때의 연결 방식(Wi-Fi/LTE)

여러 휴대폰이 있으면 휴대폰마다 3번만 다시 실행해 주세요.

## 참고: 무엇을 재나

| 항목 | 내용 |
| --- | --- |
| CLIP 임베더 다운로드 | 처음 한 번 받는 시간 (이미 받았으면 생략) |
| 첫 호출 | 모델을 메모리에 올리는 시간 포함 |
| ① MobileViT 분류 | 사진 디코드 → 전처리 → 카테고리 분류 |
| ② CLIP 임베딩 | 사진 디코드 → 전처리 → 512차원 임베딩 |
| ③ 전체 판정 | 앱의 `PhotoGate.decide` 그대로: 분류 + 임베딩 + 선형 분류기 + 유사도·문장 일치도 + 판정 |
| 메모리 | 측정 중 Java/네이티브 힙 최댓값 |

사진은 테스트가 만든 합성 사진(1080×1440, 3024×4032)을 씁니다. 속도는 사진 내용과 거의 무관합니다.
측정 코드는 `app/src/androidTest/java/smu/ai/graduation_project/data/DeviceBenchmarkTest.kt` 입니다.
