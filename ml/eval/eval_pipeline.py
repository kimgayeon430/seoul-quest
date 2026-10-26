"""논문용 평가 3단계: 판정 파이프라인 ablation + 대표 이미지/텍스트 신호 분리도.

    python ml/eval/eval_pipeline.py [--refs 1] [--split test]

실제 미션 사용자 사진은 아직 없으므로, 공개 데이터 test 셋의 **scene 을 가상 미션**으로 본다.
  - 미션 s (카테고리 c): 대표 이미지 = scene s 의 앞 k장, 미션 문구 = "a photo of a <scene 이름>"
  - 정상 제출   : 같은 scene 의 나머지 사진을 미션 s 에 제출 → PASS 가 정답
  - 다른 대상   : 같은 카테고리의 다른 scene 사진을 미션 s 에 제출 → PASS 면 오수락
  - 무효 제출   : 무효 클래스 사진을 임의 미션에 제출 → PASS 면 오수락
판정은 앱 PhotoVerification.verify 미러(common.verify), 임계값은 PhotoVerificationConfig 기본값.
대조 문구 = 다른 가상 미션 문구 전부 + embed_mission_texts.GENERIC_NEGATIVES (앱과 같은 구성).

비교 구성: A 분류만 / B +대표 이미지 유사도 / C +텍스트 일치도 / D 전부.
결과 JSON: ml/eval/results/pipeline_<split>_k<refs>.json
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from common import INVALID, ROOT, Config, clip_text_encoder, load, text_match_prob, verify

sys.path.insert(0, str(ROOT / "ml"))
from embed_mission_texts import GENERIC_NEGATIVES  # noqa: E402

CONFIGS = {"A 분류만": (False, False), "B +대표이미지": (True, False), "C +텍스트": (False, True), "D 전부": (True, True)}


def auc(pos: list[float], neg: list[float]) -> float:
    p, n = np.asarray(pos), np.asarray(neg)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    ranks = allv.argsort().argsort() + 1.0
    # 동점 평균 순위
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def prompts_for(scene: str, ensemble: bool) -> list[str]:
    name = scene.replace("_", " ")
    if not ensemble:
        return [f"a photo of a {name}"]
    return [f"a photo of a {name}", f"a photo taken at a {name}", f"a {name}"]


def build(d: dict, k: int, enc, ensemble: bool, max_wrong: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = [str(x) for x in d["labels"]]
    scenes = [str(x) for x in d["scenes"]]
    paths = [str(x) for x in d["paths"]]
    by_scene = defaultdict(list)
    for i, (l, s) in enumerate(zip(labels, scenes)):
        if l != INVALID:
            by_scene[(l, s)].append(i)
    missions = {key: sorted(v, key=lambda i: paths[i]) for key, v in by_scene.items() if len(v) >= k + 1}
    keys = sorted(missions)
    pos = {key: enc(prompts_for(key[1], ensemble)) for key in keys}
    gen = enc(GENERIC_NEGATIVES)
    neg = {key: np.concatenate([pos[o] for o in keys if o != key] + [gen]) for key in keys}
    refs = {key: d["clip_img"][missions[key][:k]] for key in keys}
    invalid_idx = [i for i, l in enumerate(labels) if l == INVALID]

    trials = []  # (scenario, mission key, image idx)
    for key in keys:
        for i in missions[key][k:]:
            trials.append(("정상", key, i))
        others = [i for o in keys if o != key and o[0] == key[0] for i in missions[o][k:]]
        for i in rng.choice(others, size=min(max_wrong, len(others)), replace=False) if others else []:
            trials.append(("다른 대상", key, int(i)))
    for i in invalid_idx:
        trials.append(("무효", keys[int(rng.integers(len(keys)))], i))
    return labels, keys, pos, neg, refs, trials


def signals(d, key, i, pos, neg, refs):
    e = d["clip_img"][i]
    sim = float((refs[key] @ e).max())
    txt = text_match_prob(e, pos[key], neg[key])
    return sim, txt


def precompute(d: dict, labels_order, keys, pos, neg, refs, trials) -> list[tuple]:
    idx = {l: j for j, l in enumerate(labels_order)}
    rows = []
    for scen, key, i in trials:
        p = d["mvit_probs"][i]
        sim, txt = signals(d, key, i, pos, neg, refs)
        rows.append((scen, float(p[idx[key[0]]]), float(p[idx[INVALID]]), sim, txt))
    return rows


def rates(rows: list[tuple], c: Config, use_sim: bool, use_txt: bool, rule: str = "or") -> dict:
    counts = defaultdict(lambda: defaultdict(int))
    for scen, m, inv, sim, txt in rows:
        counts[scen][verify(m, inv, sim if use_sim else None, txt if use_txt else None, c, rule)] += 1
    res = {s: {v: n / sum(cs.values()) for v, n in cs.items()} | {"n": sum(cs.values())} for s, cs in counts.items()}
    tot = sum(r["n"] for r in res.values())
    res["검수 비율"] = sum(r.get("REVIEW", 0) * r["n"] for r in res.values() if isinstance(r, dict) and "n" in r) / tot
    return res


def run(rows: list[tuple], c: Config, rule: str = "or") -> dict:
    return {name: rates(rows, c, us, ut, rule) for name, (us, ut) in CONFIGS.items()}


def separability(d, keys, pos, neg, refs, trials) -> dict:
    out = defaultdict(lambda: {"sim_pos": [], "sim_neg": [], "txt_pos": [], "txt_neg": []})
    for scen, key, i in trials:
        if scen == "무효":
            continue
        sim, txt = signals(d, key, i, pos, neg, refs)
        tag = "pos" if scen == "정상" else "neg"
        for cat in (key[0], "전체"):
            out[cat][f"sim_{tag}"].append(sim)
            out[cat][f"txt_{tag}"].append(txt)
    res = {}
    for cat, v in out.items():
        res[cat] = {
            "image_sim_auc": auc(v["sim_pos"], v["sim_neg"]),
            "text_match_auc": auc(v["txt_pos"], v["txt_neg"]),
            "image_sim_same_p15": float(np.percentile(v["sim_pos"], 15)), "image_sim_other_p90": float(np.percentile(v["sim_neg"], 90)),
            "text_same_p10": float(np.percentile(v["txt_pos"], 10)), "text_same_p50": float(np.median(v["txt_pos"])),
            "text_other_p95": float(np.percentile(v["txt_neg"], 95)),
            "n_pos": len(v["sim_pos"]), "n_neg": len(v["sim_neg"]),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--refs", type=int, default=1, help="가상 미션당 대표 이미지 수 k")
    ap.add_argument("--ensemble", action="store_true", help="미션 문구 3종 앙상블 (E5)")
    ap.add_argument("--max-wrong", type=int, default=10, help="미션당 '다른 대상' 제출 표본 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-suspect", type=float, default=None, help="similaritySuspectThreshold 덮어쓰기 (변경 전 0.68 재현용)")
    args = ap.parse_args()

    d = load(args.split)
    enc, _ = clip_text_encoder()
    labels_order = d["label_order"]
    labels, keys, pos, neg, refs, trials = build(d, args.refs, enc, args.ensemble, args.max_wrong, args.seed)
    c = Config() if args.sim_suspect is None else Config(simSuspect=args.sim_suspect)
    rows = precompute(d, labels_order, keys, pos, neg, refs, trials)
    abl = run(rows, c)
    sep = separability(d, keys, pos, neg, refs, trials)

    n_by = defaultdict(int)
    for s, _, _ in trials:
        n_by[s] += 1
    print(f"split={args.split} k={args.refs} ensemble={args.ensemble}  가상 미션 {len(keys)}개 "
          f"(대조 문구 {len(next(iter(neg.values())))}개)  제출: " + ", ".join(f"{k} {v}" for k, v in n_by.items()))
    print("\n구성            정상 PASS  정상 REVIEW  정상 REJECT | 다른대상 PASS(오수락) | 무효 PASS(오수락) | 전체 검수 비율")
    for name, r in abl.items():
        review = r["검수 비율"]
        print(f"{name:12s}  {r['정상'].get('PASS', 0):9.3f}  {r['정상'].get('REVIEW', 0):11.3f}  {r['정상'].get('REJECT', 0):11.3f} | "
              f"{r['다른 대상'].get('PASS', 0):20.3f} | {r['무효'].get('PASS', 0):17.3f} | {review:.3f}")
    print("\n분리도 (정상 vs 같은 카테고리 다른 대상, ROC AUC)")
    for cat, v in sep.items():
        print(f"  {cat}: 대표이미지 유사도 AUC {v['image_sim_auc']:.3f}  텍스트 일치도 AUC {v['text_match_auc']:.3f}  "
              f"(sim 정상 p15 {v['image_sim_same_p15']:.3f} / 다른 p90 {v['image_sim_other_p90']:.3f}; "
              f"text 정상 p10 {v['text_same_p10']:.3f} p50 {v['text_same_p50']:.3f} / 다른 p95 {v['text_other_p95']:.3f})")

    out = ROOT / "ml/eval/results"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"pipeline_{args.split}_k{args.refs}{'_ens' if args.ensemble else ''}_ss{c.simSuspect:.2f}"
    (out / f"{tag}.json").write_text(json.dumps(
        {"args": vars(args), "n_missions": len(keys), "n_trials": dict(n_by), "config": c.__dict__,
         "ablation": abl, "separability": sep}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
