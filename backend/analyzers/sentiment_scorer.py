from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _HAS_TX = True
except Exception:
    _HAS_TX = False


@dataclass
class SentimentBatch:
    text: List[str]
    score: List[float]   # signed valence in [-1, 1] approx (ppos - pneg)
    conf: List[float]    # confidence margin (max prob - second max)


class SentimentScorer:
    """
    Compact transformer sentiment wrapper.
    Default model: 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    Falls back to neutral when transformers is unavailable.
    """
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.enabled = _HAS_TX
        self.model_name = model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
        if not self.enabled:
            return
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.mdl.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mdl.to(self.device)

    @staticmethod
    def _valence_from_logits(logits) -> Tuple[List[float], List[float]]:
        import torch
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pneg, pneu, ppos = probs[..., 0], probs[..., 1], probs[..., 2]
        score = (ppos - pneg).tolist()
        # confidence margin: max - second max
        top = probs.max(axis=-1)
        second = np.partition(probs, -2, axis=-1)[:, -2]
        conf = (top - second).tolist()
        return score, conf

    def score(self, texts: List[str]) -> SentimentBatch:
        if not self.enabled:
            return SentimentBatch(text=texts, score=[0.0] * len(texts), conf=[0.0] * len(texts))
        import torch
        outs, confs = [], []
        for i in range(0, len(texts), 32):
            chunk = texts[i:i + 32]
            toks = self.tok(chunk, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.mdl(**toks).logits
            s, c = self._valence_from_logits(logits)
            outs.extend(s)
            confs.extend(c)
        return SentimentBatch(text=texts, score=outs, conf=confs)


def calibrate_doc(scores: List[float]) -> Tuple[float, float]:
    """
    Robust location/scale for per-doc z-scores.
    Returns (mu, sigma) where sigma ~ MAD-based scale.
    """
    x = np.array(scores, dtype=float)
    if x.size == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) + 1e-6)
    return med, 1.4826 * mad
