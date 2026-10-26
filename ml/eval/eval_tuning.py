"""개선 실험 E6: 판정 결합 규칙·임계값을 validation 가상 미션에서 고르고 test 에서 확인한다.

    python ml/eval/eval_tuning.py [--refs 1]

목표: 다른 대상 오수락률 <= --max-false-accept, 무효 오수락률 <= 0.01 를 지키면서
      정상 PASS 율을 최대화(동점이면 전체 검수 비율이 낮은 쪽). validation/test 는 scene 이 겹치지 않는다(그룹 분할).
결과 JSON: ml/eval/results/tuning_k<refs>.json
"""

import argparse
import itertools
import json

from common import ROOT, Config, clip_text_encoder, load
from eval_pipeline import build, precompute, rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", type=int, default=1)
    ap.add_argument("--max-false-accept", type=float, default=0.10)
    args = ap.parse_args()
    enc, _ = clip_text_encoder()

    def rows_for(split):
        d = load(split)
        _, keys, pos, neg, refs, trials = build(d, args.refs, enc, False, 10, 0)
        return precompute(d, d["label_order"], keys, pos, neg, refs, trials)

    val, test = rows_for("validation"), rows_for("test")
    base = Config(simSuspect=0.68)  # 이번 평가 전 앱 설정
    cands = []
    for rule, ss, tp, ts in itertools.product(
        ["or", "text_first"], [0.60, 0.65, 0.68, 0.72, 0.76, 0.80], [0.10, 0.20, 0.30, 0.40, 0.50], [0.005, 0.01, 0.02, 0.05, 0.08]
    ):
        if ts > tp:
            continue
        c = Config(simSuspect=ss, textPass=tp, textSuspect=ts)
        r = rates(val, c, True, True, rule)
        ok = r["다른 대상"].get("PASS", 0) <= args.max_false_accept and r["무효"].get("PASS", 0) <= 0.01
        cands.append((ok, r["정상"].get("PASS", 0), -r["검수 비율"], rule, c, r))
    ok = [x for x in cands if x[0]]
    best = max(ok or cands, key=lambda x: (x[1], x[2]))
    _, _, _, rule, c, rv = best

    def line(tag, r):
        return (f"{tag:28s} 정상 PASS {r['정상'].get('PASS', 0):.3f} / REJECT {r['정상'].get('REJECT', 0):.3f} | "
                f"다른대상 PASS {r['다른 대상'].get('PASS', 0):.3f} | 무효 PASS {r['무효'].get('PASS', 0):.3f} | 검수 {r['검수 비율']:.3f}")

    print(f"선택 (validation 기준, 오수락<={args.max_false_accept}): rule={rule} simSuspect={c.simSuspect} "
          f"textPass={c.textPass} textSuspect={c.textSuspect}")
    rt_base, rt_best = rates(test, base, True, True, "or"), rates(test, c, True, True, rule)
    print(line("validation 변경 전(0.68)", rates(val, base, True, True, "or")))
    print(line("validation 선택 설정", rv))
    print(line("test 변경 전(0.68)", rt_base))
    print(line("test 선택 설정", rt_best))
    out = ROOT / "ml/eval/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"tuning_k{args.refs}.json").write_text(json.dumps(
        {"args": vars(args), "chosen": {"rule": rule, **c.__dict__}, "validation_chosen": rv,
         "test_current": rt_base, "test_chosen": rt_best}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
