"""실제 미션 데이터(Firestore, 읽기 전용)로 본 대표 이미지·텍스트 신호 분포.

    python ml/eval/eval_real_missions.py --firebase-key <serviceAccount.json>

  - 대표 이미지 유사도: 같은 미션의 대표 이미지끼리 vs 다른 미션 대표 이미지 (코사인, ROC AUC)
  - 텍스트 일치도: 미션별 첫 대표 이미지 → 자기 미션 / 다른 미션 확률 (앱과 같은 photoText* 필드 사용)
쓰기는 하지 않는다. 결과: ml/eval/results/real_missions.json
"""

import argparse
import itertools
import json

import numpy as np

from common import ROOT, text_match_prob
from eval_pipeline import auc


def vecs(rows):
    out = []
    for r in rows or []:
        v = r.get("v") if isinstance(r, dict) else r
        if v:
            v = np.asarray(v, np.float32)
            out.append(v / np.linalg.norm(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--firebase-key", required=True)
    args = ap.parse_args()
    import firebase_admin
    from firebase_admin import credentials, firestore

    firebase_admin.initialize_app(credentials.Certificate(args.firebase_key))
    docs = {d.id: d.to_dict() or {} for d in firestore.client().collection("missions").stream()}
    imgs = {m: vecs(d.get("photoEmbeddings")) or vecs([d.get("photoEmbedding")]) for m, d in docs.items()}
    imgs = {m: v for m, v in imgs.items() if v}
    same = [float(a @ b) for v in imgs.values() for a, b in itertools.combinations(v, 2)]
    same_first = [float(v[0] @ b) for v in imgs.values() for b in v[1:]]
    diff = [float(imgs[m][0] @ imgs[o][0]) for m, o in itertools.permutations(imgs, 2)]

    own, other = [], []
    for m, v in imgs.items():
        d = docs[m]
        pos, neg = np.array([r["v"] for r in d.get("photoTextEmbeddings", [])], np.float32), np.array(
            [r["v"] for r in d.get("photoTextNegatives", [])], np.float32)
        if len(pos) == 0:
            continue
        own.append(text_match_prob(v[0], pos, neg, float(d.get("photoTextLogitScale", 100.0))))
        for o, d2 in docs.items():
            if o == m or not d2.get("photoTextEmbeddings"):
                continue
            p2 = np.array([r["v"] for r in d2["photoTextEmbeddings"]], np.float32)
            n2 = np.array([r["v"] for r in d2["photoTextNegatives"]], np.float32)
            other.append(text_match_prob(v[0], p2, n2, float(d2.get("photoTextLogitScale", 100.0))))
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan")
    res = {
        "missions_with_images": len(imgs), "missions_total": len(docs),
        "image_sim": {"same_mission_pairs": len(same), "same_p50": pct(same, 50), "same_p15": pct(same, 15),
                      "other_p50": pct(diff, 50), "other_p90": pct(diff, 90), "auc_all_pairs": auc(same, diff),
                      "auc_first_vs_rest": auc(same_first, diff)},
        "text_match_first_image": {"n_own": len(own), "own_p10": pct(own, 10), "own_p50": pct(own, 50),
                                   "own_top_ratio(>0.5)": float(np.mean(np.array(own) > 0.5)),
                                   "n_other": len(other), "other_p95": pct(other, 95), "other_p99": pct(other, 99),
                                   "auc": auc(own, other)},
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (ROOT / "ml/eval/results/real_missions.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
