# validation/cv.py
# ------------------------------------------------------------
# Purged KFold + Embargo, Walk-Forward Splitter (시계열 누설 방지)
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, Sequence
import numpy as np
import pandas as pd


@dataclass
class PurgedKFold:
    """
    Lopez de Prado 'Advances in Financial ML' 기반 Purged K-Fold
    - 시계열에서 label horizon/overlap로 인한 정보누설 방지
    - embargo_pct: 각 fold 경계 주변 일부 샘플 제외(검증 집합이 학습 집합의 직후에 주는 리크 방지)
    """
    n_splits: int = 5
    embargo_pct: float = 0.01

    def split(
        self,
        X: pd.DataFrame,
        t1: Optional[pd.Series] = None,
        times: Optional[pd.DatetimeIndex] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        X: index가 시계열(고정 길이)라고 가정
        t1: 각 샘플의 라벨 유효종료 시점(없으면 times와 동일 간주)
        times: 샘플 timestamp (없으면 X.index 사용)
        """
        if times is None:
            if isinstance(X.index, pd.DatetimeIndex):
                times = X.index
            else:
                raise ValueError("X.index가 DatetimeIndex가 아니면 times를 넘겨주세요.")
        if t1 is None:
            t1 = pd.Series(times, index=times)

        # 시간순으로 정렬
        order = np.argsort(times.values)
        times_sorted = times.values[order]
        t1_sorted = t1.values[order]

        n = len(times_sorted)
        fold_bounds = np.linspace(0, n, self.n_splits + 1, dtype=int)

        for i in range(self.n_splits):
            start, end = fold_bounds[i], fold_bounds[i + 1]
            test_idx = order[start:end]

            # embargo
            emb = int(np.ceil(n * max(0.0, min(0.5, self.embargo_pct))))
            # 테스트 구간의 시간창
            test_start_t = times_sorted[start]
            test_end_t = times_sorted[end - 1] if end > start else test_start_t

            # purge: 학습에서 테스트 horizon(t1)과 겹치는 것 제거
            is_overlap = (t1 >= test_start_t) & (times <= test_end_t)
            train_mask = ~is_overlap

            # embargo: 테스트 종료 이후 emb 기간 제외
            if emb > 0 and end < n:
                embargo_end_idx = min(n, end + emb)
                embargo_times = set(times_sorted[end:embargo_end_idx])
                train_mask &= ~times.isin(embargo_times)

            train_idx = np.where(train_mask.values)[0]
            yield train_idx, test_idx


@dataclass
class WalkForward:
    """
    롤링/확장 윈도우 기반 Walk-Forward 분할기
    - expanding=True: 학습 구간을 누적 확장
    - expanding=False: 고정 길이 롤링
    """
    train_size: int
    test_size: int
    step: Optional[int] = None
    expanding: bool = True

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
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
            yield train_idx, test_idx

            start_train = start_train if self.expanding else start_train + step
            if self.expanding:
                # 확장 학습은 테스트만 전진
                start_train = 0
                self.train_size = end_train  # 다음 분할은 지금까지 누적
            else:
                # 롤링 창은 고정 길이로 step 만큼 전진
                pass
            # 공통: 테스트 구간 전진
            X_slice_len_remaining = n - (start_test + step)
            if X_slice_len_remaining <= 0:
                break


def build_time_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DatetimeIndex:
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    try:
        ts = ts.tz_convert("Asia/Seoul")
    except Exception:
        pass
    if ts.isna().any():
        raise ValueError("timestamp에 NaT 존재")
    return pd.DatetimeIndex(ts)


# 간단 사용예
def example_purged_cv(df: pd.DataFrame, n_splits=5, embargo_pct=0.01):
    times = build_time_index(df)
    pkf = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    for tr, te in pkf.split(df, times=times):
        yield tr, te
