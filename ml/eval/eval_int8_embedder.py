"""앱이 실제로 쓰는 CLIP 임베더(int8 ONNX, HF Hub)로 test 임베딩을 다시 계산해 fp32 와 비교한다.

    python ml/eval/eval_int8_embedder.py

- 앱 전처리(assets/photo_embedder_preprocessor.json, bilinear 리사이즈)로 임베딩 → cache/clip_img_int8_test.npy
- fp32 임베딩과의 코사인(일치도), 그리고 E3 probe(fp32 로 학습)를 int8 임베딩에 적용했을 때 성능
결과: ml/eval/results/int8_embedder.json
"""

import io
import json

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.linear_model import LogisticRegression

from common import INVALID, ROOT, Config, acc_f1, load
from eval_classifier import report
from extract_features import ASSETS, find_parquet, resize_crop

REPO = "kimgayeon430/travel-mission-photo-embedder"


def main() -> None:
    pp = json.loads((ASSETS / "photo_embedder_preprocessor.json").read_text())
    sess = ort.InferenceSession(hf_hub_download(REPO, "photo_embedder_int8.onnx"), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    mean, std = np.array(pp["image_mean"], np.float32), np.array(pp["image_std"], np.float32)
    te = load("test")
    df = pd.read_parquet(find_parquet("test", None))
    embs = []
    for r in df.itertuples():
        img = resize_crop(Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"), pp["size"]["shortest_edge"], pp["crop_size"]["height"])
        a = (np.asarray(img, np.float32) * pp["rescale_factor"] - mean) / std
        e = sess.run(None, {name: np.ascontiguousarray(a.transpose(2, 0, 1)[None])})[0][0].astype(np.float64)
        embs.append((e / np.linalg.norm(e)).astype(np.float32))
    e8 = np.array(embs)
    np.save(ROOT / "ml/eval/cache/clip_img_int8_test.npy", e8)
    agree = (e8 * te["clip_img"]).sum(1)
    print(f"fp32 vs int8(앱) 임베딩 코사인: 평균 {agree.mean():.4f}, 최소 {agree.min():.4f}")

    # E3 probe 를 같은 방식(C=3, w=0.9)으로 재학습하고 int8 임베딩에 적용
    tr = load("train")
    res = json.loads((ROOT / "ml/eval/results/clip_probe.json").read_text())
    C, w = res["C"], res["w"]
    labels = te["label_order"]
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(tr["clip_img"], [str(x) for x in tr["labels"]])
    order = [list(clf.classes_).index(l) for l in labels]
    yte = [str(x) for x in te["labels"]]
    out = {"cos_fp32_int8_mean": float(agree.mean()), "cos_fp32_int8_min": float(agree.min())}
    inv = labels.index(INVALID)
    for tag, emb in (("fp32", te["clip_img"]), ("int8_app", e8)):
        p = clf.predict_proba(emb)[:, order]
        h = (1 - w) * te["mvit_probs"] + w * p
        h[:, inv] = te["mvit_probs"][:, inv]
        out[tag] = report(f"E3c with {tag} CLIP 임베딩", yte, h, labels, Config())
    (ROOT / "ml/eval/results/int8_embedder.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
