# ml/eval — 논문용 사진 인증·추천 재평가

논문/보고서의 수치는 모두 아래 스크립트로 재현한다. 결과 JSON 은 `results/` 에 커밋되어 있다.

```bash
cd ml/eval   # venv: torch transformers onnxruntime scikit-learn pandas pyarrow pillow huggingface_hub (+ firebase-admin)
python extract_features.py              # HF 데이터셋 d4c8a38(6,063장) → cache/{train,validation,test}.npz (약 10분, CPU)
python eval_classifier.py               # 2장: MobileViT vs CLIP zero-shot, E1 TTA, E2 zero-shot 결합
python eval_clip_probe.py               # 5장 E3: CLIP linear probe (+ --export ../../app/src/main/assets/clip_probe.json)
python eval_int8_embedder.py            # E3 를 앱 int8 임베더로 재확인
python eval_pipeline.py --sim-suspect 0.68   # 4장 ablation (변경 전)
python eval_pipeline.py                      # 4장 ablation (변경 후 0.76)
python eval_pipeline.py --probs hybrid       # 4장 최종 (probe 반영)
python eval_pipeline.py --refs 3             # 대표 이미지 3장
python eval_pipeline.py --ensemble --sim-suspect 0.68   # 미션 문구 앙상블
python eval_tuning.py                   # E4: 결합 임계값 validation 선택 → test
python eval_real_missions.py --firebase-key <serviceAccount.json>   # 3장 실제 미션(읽기 전용)
python eval_model_versions.py --old <이전 photo_verifier.onnx>      # 이전 배포 모델 비교
cd ../reco && python evaluate_reco.py --seed {0..4}                  # 추천 (시뮬레이터)
```

| 파일 | 내용 |
| --- | --- |
| `common.py` | 캐시 로드, 지표(P/R/F1·부트스트랩 CI·ECE), 앱 `PhotoVerification.verify` 파이썬 미러, 텍스트 일치도 |
| `extract_features.py` | 앱 ONNX 분류기 확률(+좌우반전), CLIP 이미지 임베딩, 추론 시간 캐시 |
| `eval_classifier.py` | 분류 성능·판정 구간·기준선·E1/E2 |
| `eval_clip_probe.py` | E3 linear probe·결합(E3b)·무효 유지 결합(E3c, 앱 반영)·McNemar |
| `eval_pipeline.py` | scene = 가상 미션. 정상/다른 대상/무효 제출의 PASS·REVIEW·REJECT, 신호 분리도 AUC |
| `eval_tuning.py` | 결합 규칙·임계값을 validation 가상 미션에서 선택 |
| `eval_real_missions.py` | Firestore 실제 미션의 대표 이미지·텍스트 신호 분포 |
| `eval_int8_embedder.py` / `eval_model_versions.py` | 앱 int8 임베더 일치도 / 이전 모델 비교 |

주의: 가상 미션의 문구는 scene 이름("a photo of a bakery")이라 실제 미션 문구보다 쉽다. 텍스트 일치도의
절대값은 실제 미션(`eval_real_missions.py`)과 함께 해석한다.
