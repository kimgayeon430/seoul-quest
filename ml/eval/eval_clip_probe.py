"""개선 실험 E3: CLIP 이미지 임베딩 위 선형 분류기(linear probe) + MobileViT 결합.

    python ml/eval/eval_clip_probe.py            # train/validation/test 캐시 필요

앱은 대표 이미지 유사도를 위해 이미 CLIP 임베딩을 기기에서 계산한다. 그 임베딩에 5클래스 로지스틱 회귀
(가중치 512×5 + 절편 5, JSON 수 KB)를 얹으면 모델을 추가하지 않고 두 번째 분류기를 얻는다.
  - C(규제)와 결합 가중치 w 는 validation macro-F1 로 고르고 test 는 한 번만 본다.
  - 결합: p = (1-w)·MobileViT + w·probe
결과 JSON: ml/eval/results/clip_probe.json (+ --export 경로에 probe 가중치: 앱 assets/clip_probe.json)
E3c(카테고리 점수만 결합, 무효 점수는 MobileViT 유지)를 앱에 반영했다(CategoryProbe).
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import CATS, INVALID, ROOT, Config, acc_f1, load
from eval_classifier import report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None, help="probe 가중치 JSON 저장 경로 (앱 반영용)")
    args = ap.parse_args()
    tr, va, te = load("train"), load("validation"), load("test")
    labels = te["label_order"]
    ytr, yva, yte = ([str(x) for x in d["labels"]] for d in (tr, va, te))

    best = None
    for C in [0.1, 0.3, 1, 3, 10, 30]:
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
        clf.fit(tr["clip_img"], ytr)
        order = [list(clf.classes_).index(l) for l in labels]
        pv = clf.predict_proba(va["clip_img"])[:, order]
        f1 = acc_f1(yva, [labels[i] for i in pv.argmax(1)], labels)[1]
        print(f"  C={C:<4} val macro-F1 {f1:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf, order)
    _, C, clf, order = best
    pv, pt = clf.predict_proba(va["clip_img"])[:, order], clf.predict_proba(te["clip_img"])[:, order]

    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    vf = [acc_f1(yva, [labels[i] for i in ((1 - w) * va["mvit_probs"] + w * pv).argmax(1)], labels)[1] for w in grid]
    w = float(grid[int(np.argmax(vf))])
    print(f"\n선택: C={C}, 결합 w={w} (val macro-F1 {max(vf):.3f}; MobileViT 단독 {vf[0]:.3f}, probe 단독 {vf[-1]:.3f})")

    c = Config()
    res = {"C": C, "w": w, "val_curve": dict(zip(grid.tolist(), vf))}
    res["mobilevit"] = report("MobileViT 단독 (기준)", yte, te["mvit_probs"], labels, c)
    res["probe"] = report(f"E3a CLIP linear probe (C={C})", yte, pt, labels, c)
    res["fusion"] = report(f"E3b MobileViT + CLIP probe (w={w})", yte, (1 - w) * te["mvit_probs"] + w * pt, labels, c)
    # E3c: 카테고리 점수만 결합하고 무효 점수는 MobileViT 그대로 (스푸핑 방어 게이트 유지)
    inv = labels.index(INVALID)
    fused = (1 - w) * te["mvit_probs"] + w * pt
    hybrid = fused.copy()
    hybrid[:, inv] = te["mvit_probs"][:, inv]
    for split, d, pp in (("validation", va, pv), ("test", te, pt)):
        h = (1 - w) * d["mvit_probs"] + w * pp
        h[:, inv] = d["mvit_probs"][:, inv]
        np.save(ROOT / f"ml/eval/cache/hybrid_probs_{split}.npy", h.astype(np.float32))
    # 정상 사진 오거절(REJECT) 쌍 비교 — 판정 수준 McNemar
    valid = np.array([t != INVALID for t in yte])
    def rej(P):
        m = P[np.arange(len(yte)), [labels.index(t) if t != INVALID else 0 for t in yte]]
        return (P[:, inv] >= c.invalidReject) | (m < c.hardReject)
    ra, rb = rej(te["mvit_probs"])[valid], rej(hybrid)[valid]
    m01, m10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
    from math import comb
    kk, nn = min(m01, m10), m01 + m10
    pr = min(1.0, 2 * sum(comb(nn, i) for i in range(kk + 1)) / 2 ** nn) if nn else 1.0
    res["mcnemar_false_reject_mvit_vs_e3c"] = {"only_mvit_rejects": m01, "only_e3c_rejects": m10, "p_exact": pr}
    print(f"오거절 McNemar (MobileViT vs E3c, 정상 {int(valid.sum())}장): MobileViT만 거절 {m01}, E3c만 거절 {m10}, 정확 p={pr:.4f}")
    res["fusion_keep_invalid"] = report(f"E3c 결합(w={w}) + 무효 점수는 MobileViT 유지", yte, hybrid, labels, c)
    # McNemar (정답 여부 불일치 쌍) — MobileViT vs E3b
    a = te["mvit_probs"].argmax(1) == np.array([labels.index(t) for t in yte])
    b = fused.argmax(1) == np.array([labels.index(t) for t in yte])
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    from math import comb
    k, n = min(n01, n10), n01 + n10
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    res["mcnemar_mvit_vs_fusion"] = {"mvit_only_correct": n01, "fusion_only_correct": n10, "p_exact": p}
    print(f"\nMcNemar (MobileViT vs E3b): MobileViT만 정답 {n01}, 결합만 정답 {n10}, 정확 p={p:.4f}")
    (ROOT / "ml/eval/results/clip_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    if args.export:
        # 앱 CategoryProbeSource 형식. weights 는 labels 순서의 행(라벨당 512), 앱은 softmax(W·e + b).
        json.dump({"version": f"clip-probe-lr-C{C}-w{w}", "embedder_version": "clip-vit-base-patch32",
                   "labels": labels, "blend": w, "keep_labels": [INVALID],
                   "weights": np.round(clf.coef_[order].astype(np.float64), 5).tolist(), "bias": np.round(clf.intercept_[order].astype(np.float64), 5).tolist()},
                  open(args.export, "w"), ensure_ascii=False, separators=(",", ":"))
        print("저장:", args.export)

if __name__ == "__main__":
    main()
