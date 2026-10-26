"""개선 실험 E3: CLIP 이미지 임베딩 위 선형 분류기(linear probe) + MobileViT 결합.

    python ml/eval/eval_clip_probe.py            # train/validation/test 캐시 필요

앱은 대표 이미지 유사도를 위해 이미 CLIP 임베딩을 기기에서 계산한다. 그 임베딩에 5클래스 로지스틱 회귀
(가중치 512×5 + 절편 5, JSON 수 KB)를 얹으면 모델을 추가하지 않고 두 번째 분류기를 얻는다.
  - C(규제)와 결합 가중치 w 는 validation macro-F1 로 고르고 test 는 한 번만 본다.
  - 결합: p = (1-w)·MobileViT + w·probe
결과 JSON: ml/eval/results/clip_probe.json (+ --export 경로에 probe 가중치)
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
    (ROOT / "ml/eval/results/clip_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    if args.export:
        coef = clf.coef_[order]
        icpt = clf.intercept_[order]
        json.dump({"version": f"clip-probe-lr-C{C}", "labels": labels, "blend": w,
                   "weights": np.round(coef, 6).tolist(), "bias": np.round(icpt, 6).tolist()},
                  open(args.export, "w"), ensure_ascii=False)
        print("저장:", args.export)


if __name__ == "__main__":
    main()
