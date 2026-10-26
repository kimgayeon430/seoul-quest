"""논문용 평가 2단계: 사진 분류 모델(MobileViT, 앱 ONNX) 재평가 + CLIP zero-shot 기준선 + 개선 실험.

    python ml/eval/eval_classifier.py            # extract_features.py 캐시 사용

출력: accuracy/macro-F1(95% bootstrap CI), 클래스별 P/R/F1, 혼동행렬, ECE, 3구간 판정 분포,
      무효 recall·정상 오거절률, 추론 시간, 그리고 개선 실험
        E1 좌우반전 TTA, E2 MobileViT + CLIP zero-shot late fusion (가중치는 validation 에서 선택).
결과 JSON: ml/eval/results/classifier.json
"""

import json

import numpy as np

from common import CATS, INVALID, ROOT, Config, acc_f1, bootstrap_ci, clip_text_encoder, ece, load, prf


def zero_shot_probs(clip_img: np.ndarray, labels: list[str], enc, scale: float) -> np.ndarray:
    """학습 노트북 2단계와 같은 방식: labels.json clip_prompts 전체 softmax → 라벨별 최대값."""
    prompts = json.loads((ROOT / "ml/labels.json").read_text())["clip_prompts"]
    flat, owner = [], []
    for i, l in enumerate(labels):
        for p in prompts[l]:
            flat.append(p)
            owner.append(i)
    t = enc(flat)
    owner = np.array(owner)
    z = clip_img @ t.T * scale
    z -= z.max(1, keepdims=True)
    s = np.exp(z)
    s /= s.sum(1, keepdims=True)
    per = np.stack([s[:, owner == i].max(1) for i in range(len(labels))], 1)
    return per / per.sum(1, keepdims=True)


def report(name: str, y: list[str], probs: np.ndarray, labels: list[str], c: Config) -> dict:
    pred = [labels[i] for i in probs.argmax(1)]
    acc, f1 = acc_f1(y, pred, labels)
    ci = bootstrap_ci(y, pred, labels)
    per = prf(y, pred, labels)
    idx = {l: i for i, l in enumerate(labels)}
    valid = [i for i, t in enumerate(y) if t != INVALID]
    inv = [i for i, t in enumerate(y) if t == INVALID]
    # 정상 사진을 "자기 카테고리 미션"에 제출했을 때의 판정 분포 (유사도·텍스트 신호 없음)
    zones = {"PASS": 0, "REVIEW": 0, "REJECT": 0}
    for i in valid:
        m, v = probs[i, idx[y[i]]], probs[i, idx[INVALID]]
        z = "REJECT" if v >= c.invalidReject else ("REJECT" if m < c.hardReject else ("PASS" if m >= c.autoPass else "REVIEW"))
        zones[z] += 1
    out = {
        "name": name,
        "accuracy": acc, "macro_f1": f1,
        "accuracy_ci95": ci["acc"].tolist(), "macro_f1_ci95": ci["f1"].tolist(),
        "per_class": {k: {"precision": v[0], "recall": v[1], "f1": v[2], "support": v[3]} for k, v in per.items()},
        "confusion": {t: {p: sum(1 for a, b in zip(y, pred) if a == t and b == p) for p in labels} for t in labels},
        "ece": ece(probs, np.array([a == b for a, b in zip(y, pred)], float)),
        "invalid_recall_at_thr": float(np.mean([probs[i, idx[INVALID]] >= c.invalidReject for i in inv])),
        "valid_zones": {k: v / len(valid) for k, v in zones.items()},
    }
    print(f"\n== {name} ==")
    print(f"  accuracy {acc:.3f} [{ci['acc'][0]:.3f}, {ci['acc'][1]:.3f}]  macro-F1 {f1:.3f} "
          f"[{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]  ECE {out['ece']:.3f}")
    print("  " + "  ".join(f"{k} P{v[0]:.3f}/R{v[1]:.3f}/F{v[2]:.3f}" for k, v in per.items()))
    print(f"  무효 recall@{c.invalidReject} {out['invalid_recall_at_thr']:.3f}  "
          f"정상 사진 판정 PASS {out['valid_zones']['PASS']:.3f} / REVIEW {out['valid_zones']['REVIEW']:.3f} / "
          f"REJECT(오거절) {out['valid_zones']['REJECT']:.3f}")
    return out


def main() -> None:
    c = Config()
    te, va = load("test"), load("validation")
    labels = te["label_order"]
    yt, yv = [str(x) for x in te["labels"]], [str(x) for x in va["labels"]]
    enc, scale = clip_text_encoder()
    zs_t, zs_v = zero_shot_probs(te["clip_img"], labels, enc, scale), zero_shot_probs(va["clip_img"], labels, enc, scale)

    res = {"n_test": len(yt), "n_validation": len(yv), "labels": labels, "config": c.__dict__}
    res["mobilevit"] = report("MobileViT-small 파인튜닝 (앱 ONNX fp32)", yt, te["mvit_probs"], labels, c)
    res["clip_zeroshot"] = report("CLIP ViT-B/32 zero-shot (labels.json 프롬프트)", yt, zs_t, labels, c)

    # E1: 좌우 반전 TTA
    tta_t = (te["mvit_probs"] + te["mvit_probs_flip"]) / 2
    res["e1_tta_flip"] = report("E1 MobileViT + 좌우반전 TTA", yt, tta_t, labels, c)

    # E2: late fusion. w 는 validation macro-F1 최대로 고른다(test 는 보지 않음).
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    vf = []
    for w in grid:
        p = (1 - w) * va["mvit_probs"] + w * zs_v
        vf.append(acc_f1(yv, [labels[i] for i in p.argmax(1)], labels)[1])
    w = float(grid[int(np.argmax(vf))])
    print(f"\nE2 fusion: validation 최적 w={w} (val macro-F1 {max(vf):.3f}, w=0 일 때 {vf[0]:.3f})")
    res["e2_fusion_w"] = w
    res["e2_fusion_val_curve"] = dict(zip(grid.tolist(), vf))
    res["e2_fusion"] = report(f"E2 MobileViT + CLIP zero-shot fusion (w={w})", yt, (1 - w) * te["mvit_probs"] + w * zs_t, labels, c)

    res["latency_ms"] = {
        "mobilevit_onnx_fp32_mac_cpu_1thread": {"p50": float(np.median(te["mvit_ms"])), "p90": float(np.percentile(te["mvit_ms"], 90))},
        "clip_vitb32_fp32_torch_mac_cpu_1thread": {"p50": float(np.median(te["clip_ms"])), "p90": float(np.percentile(te["clip_ms"], 90))},
    }
    print(f"\n추론 시간 (Mac CPU 1스레드, 배치1) MobileViT fp32 p50 {res['latency_ms']['mobilevit_onnx_fp32_mac_cpu_1thread']['p50']:.1f}ms, "
          f"CLIP fp32 p50 {res['latency_ms']['clip_vitb32_fp32_torch_mac_cpu_1thread']['p50']:.1f}ms")
    out = ROOT / "ml/eval/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "classifier.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
