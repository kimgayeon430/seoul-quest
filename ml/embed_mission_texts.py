"""미션 설명 텍스트를 CLIP 텍스트 임베딩으로 사전계산해 Firestore 에 저장한다.

    # 서비스 계정 키로 Firestore 직접 갱신 (+ 대표 이미지 기준 분포 확인)
    python embed_mission_texts.py --firebase-key serviceAccount.json --eval

    # 쓰기 없이 프롬프트와 분포만 확인
    python embed_mission_texts.py --project <firebase-project-id> --eval --dry-run

    # 키가 없으면: 미션 JSON(list[{id,title,titleEn,desc,descEn,photoPromptsEn?}]) → 결과 JSON
    python embed_mission_texts.py --missions-json missions.json --out text_embeddings.json

**왜 필요한가.** 사진 분류 모델은 "맛집 사진인가"만 보고 "*이* 미션의 대상인가"는 못 본다(보고서 6.7).
대표 이미지 유사도가 이를 보완하지만, 대표 이미지가 한 장뿐이면 각도·구도가 다른 정상 사진도
유사도가 낮게 나온다. CLIP 은 이미지와 텍스트를 같은 공간에 두므로, 미션 문구("a photo of
tteokbokki at Gwangjang Market")와 촬영본을 직접 비교할 수 있다.

**판정 방식 (앱 `PhotoEmbedding.textMatchProbOrNull`).** 촬영본 임베딩과
 - 이 미션 문구들(`photoTextEmbeddings`) 중 가장 잘 맞는 것, 그리고
 - 대조 문구들(`photoTextNegatives`: 다른 미션 문구 + 셀카·스크린샷 등 무효 문구)
의 `logitScale × 코사인` 을 softmax 에 넣어 "이 미션 문구일 확률"을 낸다. 코사인 원값(보통 0.2~0.35)은
미션·문구마다 기준이 달라 임계값을 잡기 어렵지만, 대조 문구 대비 확률은 0..1 로 비교가 쉽다.

**프롬프트 우선순위.** 미션 문서에 `photoPromptsEn`(문자열 배열, 선택)이 있으면 그것만 쓴다.
"사진에 무엇이 찍혀야 하는가"를 관리자가 영어로 적어 두면 가장 정확하다. 없으면 `titleEn`·`descEn`
으로 만들고, 영어 필드도 없으면 한국어 `title`·`desc` 를 쓰되 경고한다(openai CLIP 은 한국어에 약하다).

**모델 일치.** 앱의 이미지 임베더(`photo_embedder_int8.onnx`)와 같은 CLIP(`openai/clip-vit-base-patch32`)
이어야 한다. 텍스트 쪽은 fp32 torch 로 계산하므로 int8 이미지 쪽과 약간의 오차가 있다.

`--eval` 은 이미 저장된 대표 이미지 임베딩(`photoEmbeddings`)을 촬영본 대신 넣어
 - 자기 미션 문구에 대한 확률(높아야 함)과
 - 다른 미션 문구 기준으로 계산한 확률(낮아야 함)
의 분포를 출력한다. `PhotoVerificationConfig.textMatch*Threshold` 잠정치의 근거로 쓴다.
"""

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_ID = "openai/clip-vit-base-patch32"

# 무효 클래스(labels.json 의 "무효")와 같은 취지의 대조 문구. 미션 문구보다 이쪽에 가까우면 확률이 내려간다.
GENERIC_NEGATIVES = [
    "a selfie of a person's face",
    "a screenshot of a phone screen",
    "a photo of a computer or phone screen showing a picture",
    "a blurry photo",
    "a dark photo taken inside a pocket",
    "a photo of a ceiling or floor",
    "a photo of a random room at home",
    "a photo of a desk with a laptop",
]


def mission_prompts(d: dict) -> tuple[list[str], bool]:
    """미션 문서 → (프롬프트 목록, 영어 여부)."""
    custom = [str(p).strip() for p in (d.get("photoPromptsEn") or []) if str(p).strip()]
    if custom:
        return custom, True
    title_en = (d.get("titleEn") or "").strip()
    desc_en = (d.get("descEn") or "").strip()
    if title_en or desc_en:
        prompts = []
        if title_en:
            prompts.append(f"a photo of {title_en}")
        if desc_en:
            prompts.append(desc_en)
        return prompts, True
    title = (d.get("title") or "").strip()
    desc = (d.get("desc") or "").strip()
    return [p for p in (title, desc) if p], False


def load_clip(model_id: str):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(model_id).eval()
    tok = CLIPTokenizer.from_pretrained(model_id)
    logit_scale = float(model.logit_scale.exp().item())

    def encode(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, model.config.projection_dim), np.float32)
        batch = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        with torch.no_grad():
            out = model.get_text_features(**batch)
        # transformers 버전에 따라 텐서 대신 출력 객체를 돌려준다.
        if not isinstance(out, torch.Tensor):
            embeds = getattr(out, "text_embeds", None)
            out = embeds if embeds is not None else out.pooler_output
        e = out.cpu().numpy().astype(np.float64)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
        return e.astype(np.float32)

    return encode, logit_scale


def match_prob(img: np.ndarray, pos: np.ndarray, neg: np.ndarray, scale: float) -> float:
    """앱 `PhotoEmbedding.textMatchProbOrNull` 과 같은 계산."""
    img = img / (np.linalg.norm(img) + 1e-12)
    logits = np.concatenate([[float((pos @ img).max())], neg @ img]) * scale
    logits -= logits.max()
    p = np.exp(logits)
    return float(p[0] / p.sum())


def parse_vectors(rows) -> list[np.ndarray]:
    out = []
    for r in rows or []:
        v = r.get("v") if isinstance(r, dict) else r
        if v:
            out.append(np.asarray(v, np.float32))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID, help="HF CLIP repo id (앱 이미지 임베더와 같은 모델)")
    ap.add_argument("--firebase-key", default=None, help="서비스 계정 JSON. 주면 Firestore 를 직접 갱신")
    ap.add_argument("--project", default=None, help="Firebase 프로젝트 id (ADC 로 쓸 때)")
    ap.add_argument("--missions-json", default=None, help="키·ADC 없이 쓸 때 미션 목록 JSON")
    ap.add_argument("--out", default=None, help="결과 JSON 출력 경로")
    ap.add_argument("--version", default=None, help="photoTextModelVersion (기본: 모델 id 마지막 부분)")
    ap.add_argument("--eval", action="store_true", help="대표 이미지 임베딩으로 확률 분포 출력")
    ap.add_argument("--dry-run", action="store_true", help="Firestore 에 쓰지 않음")
    args = ap.parse_args()

    version = args.version or args.model.split("/")[-1]

    # ---- 미션 문서 확보 ----
    db = None
    docs: dict[str, dict] = {}
    if args.firebase_key or args.project:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if args.firebase_key:
            firebase_admin.initialize_app(credentials.Certificate(args.firebase_key))
        else:
            firebase_admin.initialize_app(options={"projectId": args.project})
        db = firestore.client()
        for doc in db.collection("missions").stream():
            docs[doc.id] = doc.to_dict() or {}
    elif args.missions_json:
        for d in json.loads(Path(args.missions_json).read_text(encoding="utf-8")):
            docs[str(d["id"])] = d
    else:
        raise SystemExit("--firebase-key, --project(ADC), --missions-json 중 하나가 필요합니다.")

    prompts: dict[str, list[str]] = {}
    for mid, d in docs.items():
        ps, english = mission_prompts(d)
        if not ps:
            print(f"  [skip] {mid}: 제목·설명이 없음")
            continue
        if not english:
            print(f"  [warn] {mid}: 영어 문구가 없어 한국어로 임베딩 (titleEn/descEn 또는 photoPromptsEn 권장)")
        prompts[mid] = ps
    if len(prompts) < 2:
        raise SystemExit("대조 문구를 만들려면 미션이 2개 이상 필요합니다.")

    print(f"[1/3] {args.model} 로드")
    encode, logit_scale = load_clip(args.model)
    print(f"     logit_scale={logit_scale:.2f}  미션 {len(prompts)}건\n")

    print("[2/3] 텍스트 임베딩")
    pos_emb = {mid: encode(ps) for mid, ps in prompts.items()}
    generic_emb = encode(GENERIC_NEGATIVES)
    neg_emb: dict[str, np.ndarray] = {}
    for mid in prompts:
        # 다른 미션 문구 전부 + 무효 문구. 같은 장소의 미션끼리는 서로 대조가 되어 확률이 낮아질 수 있다.
        others = [pos_emb[o] for o in prompts if o != mid]
        neg_emb[mid] = np.concatenate(others + [generic_emb])
        print(f"  {mid}: 문구 {len(prompts[mid])}개 / 대조 {len(neg_emb[mid])}개  {prompts[mid][0][:60]}")

    def rows(e: np.ndarray) -> list[dict]:
        # Firestore 는 배열의 배열을 지원하지 않으므로 embed_missions.py 와 같은 {"v": [...]} 구조로 저장한다.
        return [{"v": [round(float(x), 6) for x in v]} for v in e]

    if db is not None and not args.dry_run:
        for mid in prompts:
            db.collection("missions").document(mid).set(
                {
                    "photoTextEmbeddings": rows(pos_emb[mid]),
                    "photoTextNegatives": rows(neg_emb[mid]),
                    "photoTextLogitScale": round(logit_scale, 4),
                    "photoTextPrompts": prompts[mid],
                    "photoTextModelVersion": version,
                },
                merge=True,
            )
        print(f"\nFirestore 갱신 완료: {len(prompts)}건")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "modelVersion": version,
                    "logitScale": logit_scale,
                    "missions": {
                        mid: {
                            "prompts": prompts[mid],
                            "photoTextEmbeddings": rows(pos_emb[mid]),
                            "photoTextNegatives": rows(neg_emb[mid]),
                        }
                        for mid in prompts
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"저장: {args.out}")

    if not args.eval:
        return

    print("\n[3/3] 대표 이미지 기준 확률 분포")
    own, cross = [], []
    for mid in prompts:
        imgs = parse_vectors(docs[mid].get("photoEmbeddings")) or parse_vectors([docs[mid].get("photoEmbedding")])
        if not imgs:
            continue
        for img in imgs:
            p = match_prob(img, pos_emb[mid], neg_emb[mid], logit_scale)
            own.append(p)
            print(f"  {mid}: 자기 미션 {p:.3f}")
            for other in prompts:
                if other != mid:
                    cross.append(match_prob(img, pos_emb[other], neg_emb[other], logit_scale))
    if not own:
        print("  대표 이미지 임베딩이 있는 미션이 없습니다. embed_missions.py 를 먼저 실행하세요.")
        return
    own_a, cross_a = np.array(own), np.array(cross)
    print(f"\n  자기 미션 (n={len(own_a)}): p10 {np.percentile(own_a, 10):.3f}  "
          f"p50 {np.median(own_a):.3f}  top-1 비율 {np.mean(own_a > 0.5):.2f}")
    print(f"  다른 미션 (n={len(cross_a)}): p50 {np.median(cross_a):.3f}  p90 {np.percentile(cross_a, 90):.3f}  "
          f"p99 {np.percentile(cross_a, 99):.3f}")
    print("  → textMatchSuspectThreshold 는 '다른 미션' p90 근처, textMatchPassThreshold 는 그보다 확실히 높게 잡는다.")
    print("    대표 이미지는 현장 촬영본보다 깔끔하므로, 실사용 photoVerifyTextMatch 로 다시 확인할 것.")


if __name__ == "__main__":
    main()
