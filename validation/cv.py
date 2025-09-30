# validation/cv.py
# ------------------------------------------------------------
# Purged KFold + Embargo, Walk-Forward Splitter (시계열 누설 방지)
# + 안전장치: 단일클래스 분할 방지, 폴드 자동 축소, 최소 표본/클래스 가드
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, Sequence
import numpy as np
import pandas as pd


# ------------------------------
# 공통 유효성 검사/가드 유틸
# ------------------------------
def _enforce_time_index(X: pd.DataFrame, times: Optional[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    if times is not None:
        return times
    if isinstance(X.index, pd.DatetimeIndex):
        return X.index
    raise ValueError("X.index가 DatetimeIndex가 아니면 times를 넘겨주세요.")

def _has_enough_classes(y: Optional[Sequence], idx: np.ndarray, min_unique: int) -> bool:
    if y is None:
        return True
    if len(idx) == 0:
        return False
    vals = np.asarray(y)[idx]
    u = np.unique(vals[~pd.isna(vals)])
    return (len(u) >= min_unique)

def _has_enough_samples(idx: np.ndarray, min_samples: int) -> bool:
    return idx is not None and idx.size >= int(min_samples)

def build_time_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DatetimeIndex:
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    try:
        ts = ts.tz_convert("Asia/Seoul")
    except Exception:
        pass
    if ts.isna().any():
        raise ValueError("timestamp에 NaT 존재")
    return pd.DatetimeIndex(ts)


# ============================================================
# Purged KFold
# ============================================================
@dataclass
class PurgedKFold:
    """
    Lopez de Prado 'Advances in Financial ML' 기반 Purged K-Fold
    - 시계열에서 label horizon/overlap로 인한 정보누설 방지
    - embargo_pct: 각 fold 경계 주변 일부 샘플 제외(검증 집합이 학습 집합의 직후에 주는 리크 방지)

    추가 가드:
    - min_test_samples: 테스트 최소 표본수
    - min_train_samples: 학습 최소 표본수
    - min_unique_labels: 학습/테스트 각각 최소 고유 클래스 수(단일클래스 분할 방지)
    - auto_shrink: 유효 폴드가 부족하면 n_splits를 줄여 재시도
    """
    n_splits: int = 5
    embargo_pct: float = 0.01
    min_test_samples: int = 20
    min_train_samples: int = 50
    min_unique_labels: int = 2
    auto_shrink: bool = True
    max_shrink_steps: int = 3  # 최대 몇 번까지 폴드 수 축소 재시도할지

    def split(
        self,
        X: pd.DataFrame,
        t1: Optional[pd.Series] = None,
        times: Optional[pd.DatetimeIndex] = None,
        y: Optional[Sequence] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        X: index가 시계열(고정 길이)라고 가정
        t1: 각 샘플의 라벨 유효종료 시점(없으면 times와 동일 간주)
        times: 샘플 timestamp (없으면 X.index 사용)
        y:    (선택) 클래스 벡터. 제공되면 단일클래스/표본수 가드에 사용
        """
        times = _enforce_time_index(X, times)
        if t1 is None:
            t1 = pd.Series(times, index=times)

        # 시간순으로 정렬
        order = np.argsort(times.values)
        times_sorted = times.values[order]
        t1_sorted = t1.values[order]
        n_total = len(times_sorted)

        # 내부 함수: 현재 k로 fold 생성 + 유효성 필터
        def _yield_valid(k: int) -> Tuple[int, list[Tuple[np.ndarray, np.ndarray]]]:
            fold_bounds = np.linspace(0, n_total, k + 1, dtype=int)
            valid: list[Tuple[np.ndarray, np.ndarray]] = []
            for i in range(k):
                start, end = fold_bounds[i], fold_bounds[i + 1]
                test_idx = order[start:end]

                # embargo 길이
                emb = int(np.ceil(n_total * max(0.0, min(0.5, self.embargo_pct))))
                test_start_t = times_sorted[start]
                test_end_t = times_sorted[end - 1] if end > start else test_start_t

                # purge: 학습에서 테스트 horizon(t1)과 겹치는 것 제거
                is_overlap = (t1 >= test_start_t) & (times <= test_end_t)
                train_mask = ~is_overlap

                # embargo: 테스트 종료 이후 emb 기간 제외
                if emb > 0 and end < n_total:
                    embargo_end_idx = min(n_total, end + emb)
                    embargo_times = set(times_sorted[end:embargo_end_idx])
                    train_mask &= ~times.isin(embargo_times)

                train_idx = np.where(train_mask.values)[0]

                # --------- 유효성 가드 ----------
                if not _has_enough_samples(test_idx, self.min_test_samples):
                    continue
                if not _has_enough_samples(train_idx, self.min_train_samples):
                    continue
                if not _has_enough_classes(y, train_idx, self.min_unique_labels):
                    continue
                if not _has_enough_classes(y, test_idx, self.min_unique_labels):
                    continue

                valid.append((train_idx, test_idx))
            return len(valid), valid

        # 1차 시도
        k = int(self.n_splits)
        got, valid_pairs = _yield_valid(k)

        # 필요 시 폴드 자동 축소
        shrink_steps = 0
        while self.auto_shrink and got == 0 and k > 2 and shrink_steps < self.max_shrink_steps:
            k -= 1
            shrink_steps += 1
            got, valid_pairs = _yield_valid(k)

        # 마지막으로 유효한 폴드만 방출
        for tr, te in valid_pairs:
            yield tr, te


# ============================================================
# Walk-Forward Splitter
# ============================================================
@dataclass
class WalkForward:
    """
    롤링/확장 윈도우 기반 Walk-Forward 분할기
    - expanding=True: 학습 구간을 누적 확장
    - expanding=False: 고정 길이 롤링

    추가 가드:
    - min_test_samples / min_train_samples
    - min_unique_labels (단일클래스 방지)
    """
    train_size: int
    test_size: int
    step: Optional[int] = None
    expanding: bool = True
    min_test_samples: int = 20
    min_train_samples: int = 50
    min_unique_labels: int = 2

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[Sequence] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        step = int(self.step or self.test_size)
        start_train = 0

        while True:
            end_train = start_train + self.train_size
            start_test = end_train
            end_test = start_test + self.test_size
            if end_test > n:
                break

            train_idx = np.arange(0, end_train) if self.expanding else np.arange(start_train, end_train)
            test_idx = np.arange(start_test, end_test)

            # 유효성 가드
            if _has_enough_samples(test_idx, self.min_test_samples) \
               and _has_enough_samples(train_idx, self.min_train_samples) \
               and _has_enough_classes(y, train_idx, self.min_unique_labels) \
               and _has_enough_classes(y, test_idx, self.min_unique_labels):
                yield train_idx, test_idx

            # 창 전진
            if self.expanding:
                start_train = 0
                self.train_size = end_train  # 누적 확장
            else:
                start_train = start_train + step

            # 다음 테스트 시작점 기준 남은 길이 확인
            if (start_test + step) >= n:
                break


# 간단 사용예(유지)
def example_purged_cv(df: pd.DataFrame, n_splits=5, embargo_pct=0.01):
    times = build_time_index(df)
    pkf = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    for tr, te in pkf.split(df, times=times):
        yield tr, te
