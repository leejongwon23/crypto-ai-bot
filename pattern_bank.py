# pattern_bank.py
# v1: 간단 KNN 인덱스 (표준화 + cosine/euclidean) 구축/저장/질의
from __future__ import annotations
import os, json, time, math, hashlib
from typing import List, Dict, Any, Optional, Tuple, Iterable
import numpy as np

PERSIST_DIR = "/persistent"
BANK_DIR    = os.path.join(PERSIST_DIR, "pattern_bank")
os.makedirs(BANK_DIR, exist_ok=True)

def _now_ts() -> str:
    import datetime, pytz
    return datetime.datetime.now(pytz.timezone("Asia/Seoul")).isoformat()

def _to_2d(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a

def _safe_topk(k: int, n: int) -> int:
    k = int(k)
    if k <= 0: return 1
    return min(k, max(1, n))

class PatternBank:
    """
    간단한 KNN 인덱스:
      - metric: 'cosine' 또는 'euclidean'
      - 표준화(z-score) 지원 (fit 시점에 mean/std 저장)
      - 저장/로드: /persistent/pattern_bank/{name}/*
    """
    def __init__(self, name: str = "default", metric: str = "cosine", standardize: bool = True):
        metric = metric.lower().strip()
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.name = name
        self.metric = metric
        self.standardize = bool(standardize)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.X: Optional[np.ndarray] = None  # 저장 시: 표준화(및 정규화) 후 벡터
        self.meta: List[Dict[str, Any]] = []

    # ---------- fit / add ----------
    def fit(self, X: Iterable, meta: Optional[Iterable[Dict[str, Any]]] = None):
        X = _to_2d(X)
        if self.standardize:
            self.mean_ = X.mean(axis=0, keepdims=True)
            self.std_  = X.std(axis=0, keepdims=True) + 1e-8
            Xs = (X - self.mean_) / self.std_
        else:
            Xs = X.copy()

        if self.metric == "cosine":
            # 코사인은 크기 영향 제거 위해 L2 normalize
            nrm = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12
            Xs = Xs / nrm

        self.X = Xs.astype(np.float32)
        self.meta = list(meta) if meta is not None else [{} for _ in range(self.X.shape[0])]
        return self

    def add(self, X_new: Iterable, meta_new: Optional[Iterable[Dict[str, Any]]] = None):
        assert self.X is not None, "call fit() before add()"
        X_new = _to_2d(X_new)
        # 기존 표준화 파라미터 유지
        if self.standardize and (self.mean_ is not None):
            Xs = (X_new - self.mean_) / (self.std_ if self.std_ is not None else 1.0)
        else:
            Xs = X_new.copy()
        if self.metric == "cosine":
            nrm = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12
            Xs = Xs / nrm
        self.X = np.vstack([self.X, Xs.astype(np.float32)])
        if meta_new is None:
            meta_new = [{} for _ in range(Xs.shape[0])]
        self.meta.extend(list(meta_new))

    # ---------- query ----------
    def _pairwise_dist(self, q: np.ndarray) -> np.ndarray:
        # q: (m, d), X: (n, d)
        assert self.X is not None, "index is empty"
        if self.metric == "euclidean":
            # (q - X)^2 = q^2 + X^2 - 2 qX
            q2 = (q * q).sum(axis=1, keepdims=True)          # (m,1)
            x2 = (self.X * self.X).sum(axis=1, keepdims=True).T  # (1,n)
            d2 = q2 + x2 - 2.0 * q.dot(self.X.T)
            d2 = np.maximum(d2, 0.0)
            return np.sqrt(d2, dtype=np.float32)
        else:  # cosine distance = 1 - cosine similarity
            # X, q 둘 다 L2 normalized 상태이어야 함
            sim = q.dot(self.X.T)  # (m,n)
            return (1.0 - sim).astype(np.float32)

    def _prep_query(self, q: Iterable) -> np.ndarray:
        q = _to_2d(q).astype(float)
        if self.standardize and (self.mean_ is not None):
            q = (q - self.mean_) / (self.std_ if self.std_ is not None else 1.0)
        if self.metric == "cosine":
            nrm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
            q = q / nrm
        return q.astype(np.float32)

    def query(self, q: Iterable, topk: int = 20) -> List[Dict[str, Any]]:
        q = self._prep_query(q)
        D = self._pairwise_dist(q)  # (m, n)
        d = D[0]  # 단건 질의 가정
        k = _safe_topk(topk, d.shape[0])
        idx = np.argpartition(d, k-1)[:k]  # top-k (partial sort)
        # 정렬
        idx = idx[np.argsort(d[idx])]
        out = []
        for i in idx:
            r = {
                "index": int(i),
                "dist": float(d[i]),
                "meta": self.meta[i] if i < len(self.meta) else {}
            }
            out.append(r)
        return out

    def query_batch(self, Q: Iterable, topk: int = 20) -> List[List[Dict[str, Any]]]:
        Q = self._prep_query(Q)
        D = self._pairwise_dist(Q)  # (m, n)
        m, n = D.shape
        out_all = []
        for j in range(m):
            d = D[j]
            k = _safe_topk(topk, n)
            idx = np.argpartition(d, k-1)[:k]
            idx = idx[np.argsort(d[idx])]
            out = [{"index": int(i), "dist": float(d[i]), "meta": self.meta[i] if i < len(self.meta) else {}} for i in idx]
            out_all.append(out)
        return out_all

    # ---------- save / load ----------
    def save(self, dirpath: Optional[str] = None):
        if dirpath is None: dirpath = os.path.join(BANK_DIR, self.name)
        os.makedirs(dirpath, exist_ok=True)
        info = {
            "name": self.name,
            "metric": self.metric,
            "standardize": self.standardize,
            "created_at": _now_ts(),
            "n": int(self.X.shape[0]) if self.X is not None else 0,
            "d": int(self.X.shape[1]) if self.X is not None else 0,
        }
        if self.mean_ is not None: np.save(os.path.join(dirpath, "mean.npy"), self.mean_)
        if self.std_  is not None: np.save(os.path.join(dirpath, "std.npy"),  self.std_)
        if self.X     is not None: np.save(os.path.join(dirpath, "vectors.npy"), self.X)
        with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)
        with open(os.path.join(dirpath, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
        return dirpath

    @classmethod
    def load(cls, name: str = "default", dirpath: Optional[str] = None) -> "PatternBank":
        if dirpath is None: dirpath = os.path.join(BANK_DIR, name)
        with open(os.path.join(dirpath, "info.json"), "r", encoding="utf-8") as f:
            info = json.load(f)
        bank = cls(name=info.get("name", name),
                   metric=info.get("metric", "cosine"),
                   standardize=bool(info.get("standardize", True)))
        mean_p = os.path.join(dirpath, "mean.npy")
        std_p  = os.path.join(dirpath, "std.npy")
        vec_p  = os.path.join(dirpath, "vectors.npy")
        meta_p = os.path.join(dirpath, "meta.json")
        if os.path.exists(mean_p): bank.mean_ = np.load(mean_p)
        if os.path.exists(std_p):  bank.std_  = np.load(std_p)
        if os.path.exists(vec_p):  bank.X     = np.load(vec_p)
        if os.path.exists(meta_p):
            with open(meta_p, "r", encoding="utf-8") as f:
                bank.meta = json.load(f)
        else:
            bank.meta = [{} for _ in range(bank.X.shape[0] if bank.X is not None else 0)]
        return bank

# --------- 편의 함수들 ---------
def build_index(name: str, vectors: Iterable, meta: Optional[Iterable[Dict[str, Any]]] = None,
                metric: str = "cosine", standardize: bool = True) -> str:
    bank = PatternBank(name=name, metric=metric, standardize=standardize).fit(vectors, meta=meta)
    return bank.save()

def load_index(name: str) -> PatternBank:
    return PatternBank.load(name=name)

def query_index(name: str, qvec: Iterable, topk: int = 20) -> List[Dict[str, Any]]:
    bank = load_index(name)
    return bank.query(qvec, topk=topk)

# 간단 CLI 테스트:  python -m pattern_bank build|query ...
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "query"])
    ap.add_argument("--name", default="default")
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    if args.cmd == "build":
        # 예시: 임의 데이터 100x16
        X = np.random.randn(100, 16)
        meta = [{"idx": i} for i in range(100)]
        path = build_index(args.name, X, meta, metric=args.metric)
        print("saved to:", path)
    else:
        bank = load_index(args.name)
        q = np.random.randn(bank.X.shape[1]) if (bank.X is not None) else np.zeros(16)
        res = bank.query(q, topk=args.topk)
        print(json.dumps(res, ensure_ascii=False, indent=2))
