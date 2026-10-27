"""논문용 평가 1단계: test/validation 이미지마다 모델 출력을 한 번만 계산해 캐시한다.

    python ml/eval/extract_features.py --out ml/eval/cache

저장 (split 마다 `<split>.npz`):
  - labels / scenes / paths : 정답 클래스명, sceneId(파일명 `<scene>__n.jpg` 앞부분), 원본 경로
  - mvit_probs  [N,5] : 앱 `photo_verifier.onnx` (MobileViT, fp32 20MB) softmax. 라벨 순서는 assets 의 labels json
  - mvit_probs_flip   : 좌우 반전 입력 (TTA 실험용)
  - clip_img    [N,512] : CLIP ViT-B/32 이미지 임베딩(L2 정규화). 앱 임베더와 같은 모델의 fp32 버전
  - mvit_ms / clip_ms : 이미지당 추론 시간(ms, 이 Mac CPU, 배치 1)

데이터: HF `kimgayeon430/travel-mission-photos` @ REVISION (배포 모델 학습과 같은 분할). --data-dir 를 주면 그 parquet 을 읽는다.
전처리는 assets 의 *_preprocessor.json 을 그대로 따른다(앱 `OnnxPhotoVerifier` 와 동일한 상수).
"""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "app/src/main/assets"
DATASET = "kimgayeon430/travel-mission-photos"
# 배포 모델(mobilevit-small-fullft-1)을 학습한 6,063장 재구축본(recapture 포함). 이전 5,300장 리비전과 분할이 다르다.
REVISION = "d4c8a38ac0663a4f1231fdc1c1b1aac788070c7a"
# 데이터셋 label id → 이름 (README 의 class_label 순서, 가나다순)
DATASET_NAMES = ["맛집", "무효", "쇼핑", "체험", "투어"]


def find_parquet(split: str, data_dir: str | None) -> Path:
    if data_dir:
        return Path(data_dir) / f"{split}-00000-of-00001.parquet"
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(DATASET, f"data/{split}-00000-of-00001.parquet", repo_type="dataset", revision=REVISION))


def resize_crop(img: Image.Image, shortest: int, crop: int) -> Image.Image:
    w, h = img.size
    s = shortest / min(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - crop) // 2, (h - crop) // 2
    return img.crop((l, t, l + crop, t + crop))


def mvit_input(img: Image.Image, pp: dict) -> np.ndarray:
    x = resize_crop(img, pp["size"]["shortest_edge"], pp["crop_size"]["height"])
    a = np.asarray(x, np.float32) * pp["rescale_factor"]
    if pp.get("do_flip_channel_order"):
        a = a[..., ::-1]
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "ml/eval/cache"))
    ap.add_argument("--splits", default="test,validation,train")
    ap.add_argument("--data-dir", default=None, help="parquet 이 있는 폴더 (없으면 HF 캐시/다운로드)")
    args = ap.parse_args()

    import onnxruntime as ort
    import torch
    from transformers import CLIPModel, CLIPProcessor

    labels = json.loads((ASSETS / "photo_verifier_labels.json").read_text())["labels"]
    mpp = json.loads((ASSETS / "photo_verifier_preprocessor.json").read_text())
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(ASSETS / "photo_verifier.onnx"), so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    torch.set_num_threads(1)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        df = pd.read_parquet(find_parquet(split, args.data_dir))
        n = len(df)
        probs, probs_f, embs, mms, cms, paths = [], [], [], [], [], []
        for i, row in enumerate(df.itertuples()):
            img = Image.open(io.BytesIO(row.image["bytes"])).convert("RGB")
            paths.append(row.image["path"])
            t0 = time.perf_counter()
            logits = sess.run([out_name], {in_name: mvit_input(img, mpp)})[0][0]
            mms.append((time.perf_counter() - t0) * 1000)
            probs.append(softmax(logits))
            lf = sess.run([out_name], {in_name: mvit_input(img.transpose(Image.FLIP_LEFT_RIGHT), mpp)})[0][0]
            probs_f.append(softmax(lf))
            t0 = time.perf_counter()
            with torch.no_grad():
                e = clip.get_image_features(**proc(images=img, return_tensors="pt"))
            if not isinstance(e, torch.Tensor):
                e = getattr(e, "image_embeds", None) if getattr(e, "image_embeds", None) is not None else e.pooler_output
            cms.append((time.perf_counter() - t0) * 1000)
            e = e[0].numpy().astype(np.float64)
            embs.append((e / np.linalg.norm(e)).astype(np.float32))
            if (i + 1) % 100 == 0:
                print(f"  {split} {i + 1}/{n}")
        np.savez_compressed(
            out / f"{split}.npz",
            labels=np.array([DATASET_NAMES[int(l)] for l in df["label"]]),
            scenes=np.array([p.split("__")[0] for p in paths]),
            paths=np.array(paths),
            label_order=np.array(labels),
            mvit_probs=np.array(probs, np.float32),
            mvit_probs_flip=np.array(probs_f, np.float32),
            clip_img=np.array(embs, np.float32),
            mvit_ms=np.array(mms, np.float32),
            clip_ms=np.array(cms, np.float32),
        )
        print(f"[ok] {split}: {n}장 → {out / (split + '.npz')}")


if __name__ == "__main__":
    main()
