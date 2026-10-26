"""모델 버전 비교: 이전 배포 모델(--old, 화면 재촬영 합성 학습 전)과 현재 배포 모델을 같은 test 셋에서 비교한다.

    python ml/eval/eval_model_versions.py --old <이전 photo_verifier.onnx (+ .onnx.data)>

이전 모델은 저장소에 없다(2026-09-08 export, git 이력 또는 photo_verifier_assets.zip).
현재 모델 결과는 extract_features.py 캐시(test.npz)를 쓴다. 결과: ml/eval/results/model_versions.json
주의: 이전 모델은 5,300장 리비전으로 학습돼, 현재 test 의 무효(합성) 사진 일부가 그 학습셋과 겹쳤을 수 있다
(공개 데이터 4클래스는 scene 분할이 같아 겹치지 않음).
"""

import argparse
import io
import json

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from common import INVALID, ROOT, acc_f1, load, prf
from extract_features import ASSETS, DATASET_NAMES, find_parquet, mvit_input, softmax


def summarize(y, probs, labels):
    pred = [labels[k] for k in probs.argmax(1)]
    acc, f1 = acc_f1(y, pred, labels)
    inv = [k for k, t in enumerate(y) if t == INVALID]
    return {"accuracy": acc, "macro_f1": f1, "per_class_f1": {k: v[2] for k, v in prf(y, pred, labels).items()},
            "invalid_recall_at_0.55": float(np.mean(probs[inv, labels.index(INVALID)] >= 0.55))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    te = load("test")
    labels = te["label_order"]
    pp = json.loads((ASSETS / "photo_verifier_preprocessor.json").read_text())
    s = ort.InferenceSession(args.old, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    df = pd.read_parquet(find_parquet("test", args.data_dir))
    probs = np.array([softmax(s.run(None, {name: mvit_input(Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"), pp)})[0][0])
                      for r in df.itertuples()])
    y = [DATASET_NAMES[int(l)] for l in df["label"]]
    res = {"previous_0908": summarize(y, probs, labels), "current_fullft1": summarize([str(x) for x in te["labels"]], te["mvit_probs"], labels)}
    for k, v in res.items():
        print(f"{k:15s} acc {v['accuracy']:.3f}  macro-F1 {v['macro_f1']:.3f}  무효 recall@0.55 {v['invalid_recall_at_0.55']:.3f}  "
              + " ".join(f"{c}{f:.3f}" for c, f in v["per_class_f1"].items()))
    (ROOT / "ml/eval/results/model_versions.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
