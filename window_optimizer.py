# === window_optimizer.py (최종본) ===
import numpy as np

# ──────────────────────────────────────────────────────────────
# 단일 최적 윈도우 탐색
# ──────────────────────────────────────────────────────────────
def find_best_window(X, y, min_window=10, max_window=120, step=5):
    """
    피처/레이블 데이터를 기반으로 최적 윈도우 크기 탐색.
    min_window ~ max_window 범위에서 step 간격으로 검증.
    """
    try:
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            print("[find_best_window] 데이터 없음")
            return None

        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)

        best_window = min_window
        best_score = -np.inf

        # 윈도우별 검증
        for window in range(min_window, max_window + 1, step):
            try:
                if len(X) < window:
                    continue

                # 단순 평가 지표: 최근 구간 변동성 + 라벨 변화율
                recent_vol = np.std(X[-window:])
                label_change = np.mean(np.diff(y[-window:]) != 0)

                score = recent_vol * (1 + label_change)

                if np.isnan(score) or np.isinf(score):
                    continue

                if score > best_score:
                    best_score = score
                    best_window = window

            except Exception as e:
                print(f"[find_best_window] 내부 루프 예외: {e}")
                continue

        return best_window

    except Exception as e:
        print(f"[find_best_window] 전체 예외: {e}")
        return None

# ──────────────────────────────────────────────────────────────
# 복수 윈도우 탐색
# ──────────────────────────────────────────────────────────────
def find_best_windows(X, y, num_windows=3, min_window=10, max_window=120, step=5):
    """
    상위 num_windows 개의 윈도우 크기 반환.
    """
    try:
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            print("[find_best_windows] 데이터 없음")
            return []

        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)

        scores = []
        for window in range(min_window, max_window + 1, step):
            try:
                if len(X) < window:
                    continue

                recent_vol = np.std(X[-window:])
                label_change = np.mean(np.diff(y[-window:]) != 0)
                score = recent_vol * (1 + label_change)

                if np.isnan(score) or np.isinf(score):
                    continue

                scores.append((window, score))

            except Exception as e:
                print(f"[find_best_windows] 내부 루프 예외: {e}")
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scores[:num_windows]]

    except Exception as e:
        print(f"[find_best_windows] 전체 예외: {e}")
        return []
