"""ml/eval 스크립트 공용: 캐시 로드, 지표, 앱 판정 규칙의 파이썬 미러."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "ml/eval/cache"
INVALID = "무효"
CATS = ["투어", "맛집", "체험", "쇼핑"]
CLIP_MODEL = "openai/clip-vit-base-patch32"


def load(split: str) -> dict:
    d = dict(np.load(CACHE / f"{split}.npz"))
    d["label_order"] = [str(x) for x in d["label_order"]]
    return d


def prf(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    out = {}
    for c in labels:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[c] = (pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0, tp + fn)
    return out


def acc_f1(y_true, y_pred, labels) -> tuple[float, float]:
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    return acc, float(np.mean([v[2] for v in prf(y_true, y_pred, labels).values()]))


def bootstrap_ci(y_true, y_pred, labels, n=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    yt, yp = np.array(y_true), np.array(y_pred)
    accs, f1s = [], []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        a, f = acc_f1(yt[i].tolist(), yp[i].tolist(), labels)
        accs.append(a)
        f1s.append(f)
    return {"acc": np.percentile(accs, [2.5, 97.5]), "f1": np.percentile(f1s, [2.5, 97.5])}


def ece(probs: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


@dataclass
class Config:
    """app/.../domain/PhotoVerificationConfig.kt 기본값. simSuspect 는 이번 평가로 0.68 → 0.76 (eval_tuning.py)."""
    autoPass: float = 0.65
    hardReject: float = 0.22
    invalidReject: float = 0.55
    simRescue: float = 0.50
    simSuspect: float = 0.76
    textPass: float = 0.30
    textSuspect: float = 0.02


def verify(match: float, invalid: float, sim: float | None, text: float | None, c: Config = Config(),
           rule: str = "or") -> str:
    """app/.../domain/PhotoVerification.verify 와 같은 판정(rule="or"). PASS / REVIEW / REJECT.

    rule="text_first" (개선 실험 E6): 텍스트 일치도가 있으면 통과/의심 근거는 텍스트로만 보고,
    대표 이미지 유사도는 REJECT 구제에만 쓴다.
    """
    if invalid >= c.invalidReject:
        return "REJECT"
    base = "REJECT" if match < c.hardReject else ("PASS" if match >= c.autoPass else "REVIEW")
    if sim is None and text is None:
        return base
    sim_ge = lambda t: sim is not None and sim >= t
    txt_ge = lambda t: text is not None and text >= t
    rescue = sim_ge(c.simRescue) or txt_ge(c.textPass)
    support = sim_ge(c.simSuspect) or txt_ge(c.textPass)
    against = not sim_ge(c.simSuspect) and not txt_ge(c.textSuspect)
    if rule == "text_first" and text is not None:
        support = txt_ge(c.textPass)
        against = not txt_ge(c.textSuspect)
    if base == "REJECT" and rescue:
        return "REVIEW"
    if base == "REVIEW" and support:
        return "PASS"
    if base == "PASS" and against:
        return "REVIEW"
    return base


def text_match_prob(img: np.ndarray, pos: np.ndarray, neg: np.ndarray, scale: float = 100.0) -> float:
    """app PhotoEmbedding.textMatchProbOrNull / ml/embed_mission_texts.match_prob 와 같은 계산."""
    logits = np.concatenate([[float((pos @ img).max())], neg @ img]) * scale
    logits -= logits.max()
    p = np.exp(logits)
    return float(p[0] / p.sum())


def clip_text_encoder():
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)

    def enc(texts: list[str]) -> np.ndarray:
        b = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        with torch.no_grad():
            o = model.get_text_features(**b)
        if not isinstance(o, torch.Tensor):
            o = o.text_embeds if getattr(o, "text_embeds", None) is not None else o.pooler_output
        e = o.numpy().astype(np.float64)
        return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)

    return enc, float(model.logit_scale.exp())
