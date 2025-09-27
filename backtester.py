# backtester.py
# ------------------------------------------------------------
# 경량 백테스트 엔진 (수수료/슬리피지/펀딩 반영, 단일 심볼/전략)
# ------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class CostConfig:
    taker_fee: float = 0.0006      # 거래 수수료 (왕복은 *2가 보통)
    slippage: float = 0.0005       # 체결 슬리피지 (양방향 합산 가정)
    funding_8h: float = 0.0001     # 8시간 펀딩비(절댓값 평균) 가정
    leverage: float = 3.0          # 레버리지 (펀딩/수수료는 명목치에 반영됨)
    funding_interval_hours: int = 8


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    summary: Dict[str, float]


def _to_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)


def compute_metrics(equity: pd.Series) -> Dict[str, float]:
    ret = equity.pct_change().fillna(0.0)
    cum = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    dd = (equity / equity.cummax()).fillna(1.0) - 1.0
    mdd = float(dd.min())
    ann = 365 * 24  # 시간봉과 무관하게 충분히 큰 분모 대비 → 보수적
    vol = float(ret.std() * np.sqrt(ann))
    shrp = float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(ann)) if ret.std() > 0 else 0.0
    return {"CAGR_like": cum, "MDD": mdd, "Sharpe_like": shrp, "Vol_like": vol}


def _class_to_direction(c: int) -> int:
    # 예: 0=강한 하락, 1=중립, 2=강한 상승 (프로젝트 클래스 체계에 맞게 조정)
    if c <= 0:
        return -1
    if c >= 2:
        return +1
    return 0


def _apply_cost(price: float, side: int, cfg: CostConfig) -> float:
    """
    체결가격에 수수료/슬리피지 반영 (보수적으로 불리하게)
    """
    fee = price * cfg.taker_fee
    slp = price * cfg.slippage
    if side > 0:  # long 진입: 더 비싼 가격에 체결
        return price * (1 + cfg.slippage) + fee
    elif side < 0:  # short 진입: 더 낮은 가격에 체결
        return price * (1 - cfg.slippage) - fee
    else:
        return price


def _estimate_funding_cost(hours_held: float, notional: float, cfg: CostConfig) -> float:
    if hours_held <= 0:
        return 0.0
    k = hours_held / float(cfg.funding_interval_hours)
    # 절댓값 기준 평균 펀딩을 비용으로 가정(롱/숏 동일 불리)
    return notional * cfg.funding_8h * k


@dataclass
class Backtester:
    cfg: CostConfig = field(default_factory=CostConfig)

    def run(
        self,
        df: pd.DataFrame,
        preds: List[int],
        price_col: str = "close",
        ts_col: str = "timestamp",
        capital: float = 1_000.0,
        risk_per_trade: float = 1.0,   # 전액 진입(=1.0); 부분배팅도 허용
    ) -> BacktestResult:
        """
        df: 시계열 OHLCV 포함 DataFrame (timestamp 반드시 필요)
        preds: 각 시점 예측 클래스(list/array), df 길이와 동일하다고 가정
        """
        if price_col not in df.columns:
            raise ValueError(f"{price_col} 컬럼이 필요합니다.")
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        try:
            ts = ts.tz_convert("Asia/Seoul")
        except Exception:
            pass
        px = pd.to_numeric(df[price_col], errors="coerce").astype(float).values
        if len(preds) != len(px):
            raise ValueError("preds 길이는 df 길이와 같아야 합니다.")

        equity = [capital]
        time_axis = [ts.iloc[0]]
        pos_side = 0           # -1, 0, +1
        pos_entry = 0.0
        pos_time = ts.iloc[0]
        size = 0.0
        trades = []

        for i in range(1, len(px)):
            now_t = ts.iloc[i]
            now_p = float(px[i])
            pred_c = int(preds[i])
            desired = _class_to_direction(pred_c)

            # 포지션 유지/청산 로직 간단화: 예측 방향이 반대로 바뀌면 청산 후 재진입
            if pos_side != 0 and desired != pos_side:
                # 청산
                exit_p = _apply_cost(now_p, -pos_side, self.cfg)
                pnl = (exit_p - pos_entry) * pos_side * size * self.cfg.leverage
                hours_held = (now_t - pos_time).total_seconds() / 3600.0
                funding = _estimate_funding_cost(hours_held, pos_entry * size * self.cfg.leverage, self.cfg)
                pnl -= funding
                capital += pnl
                trades.append({
                    "entry_time": pos_time, "exit_time": now_t,
                    "side": pos_side, "entry": pos_entry, "exit": exit_p,
                    "size": size, "pnl": pnl, "funding": -funding
                })
                pos_side = 0
                size = 0.0

            # 진입/유지
            if desired != 0 and pos_side == 0:
                # 신규 진입
                entry_p = _apply_cost(now_p, desired, self.cfg)
                notional = capital * risk_per_trade
                size = notional / max(1e-12, entry_p)
                pos_entry = entry_p
                pos_side = desired
                pos_time = now_t

            # 지분가치 평가(미실현 포함)
            if pos_side != 0:
                mtm = (now_p - pos_entry) * pos_side * size * self.cfg.leverage
                # 슬리피지는 체결시에만 반영, 평가손익에는 미반영
                equity.append(capital + mtm)
            else:
                equity.append(capital)

            time_axis.append(now_t)

        # 마지막 캔들에서 열린 포지션 정리
        if pos_side != 0:
            exit_p = _apply_cost(px[-1], -pos_side, self.cfg)
            pnl = (exit_p - pos_entry) * pos_side * size * self.cfg.leverage
            hours_held = (ts.iloc[-1] - pos_time).total_seconds() / 3600.0
            funding = _estimate_funding_cost(hours_held, pos_entry * size * self.cfg.leverage, self.cfg)
            pnl -= funding
            capital += pnl
            trades.append({
                "entry_time": pos_time, "exit_time": ts.iloc[-1],
                "side": pos_side, "entry": pos_entry, "exit": exit_p,
                "size": size, "pnl": pnl, "funding": -funding
            })

        eq = pd.Series(equity, index=pd.DatetimeIndex(time_axis), name="equity")
        tr = pd.DataFrame(trades)
        metrics = compute_metrics(eq)

        if not tr.empty:
            win = float((tr["pnl"] > 0).mean())
            metrics.update({
                "WinRate": win,
                "AvgPnL": float(tr["pnl"].mean()),
                "MedianPnL": float(tr["pnl"].median()),
                "Trades": int(len(tr)),
                "Fees&Slippage_Apx": float(
                    abs(tr["entry"] * tr["size"]).sum() * self.cfg.taker_fee
                    + abs(tr["exit"] * tr["size"]).sum() * self.cfg.taker_fee
                ),
                "Funding_Apx": float(-tr["funding"].sum())
            })
        return BacktestResult(eq, tr, metrics)
