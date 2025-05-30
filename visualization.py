import pandas as pd
import matplotlib.pyplot as plt
import io, base64, numpy as np
from datetime import datetime
import pytz
import os
from matplotlib import font_manager

# ✅ 한글 + 이모지 폰트 적용
font_path = os.path.join("fonts", "NanumGothic-Regular.ttf")
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)

plt.rcParams['font.family'] = ['NanumGothic', 'Noto Color Emoji']  # ✅ 이모지 폰트 추가
plt.rcParams['axes.unicode_minus'] = False

PREDICTION_LOG = "/persistent/prediction_log.csv"
AUDIT_LOG = "/persistent/logs/evaluation_audit.csv"

def load_df(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Seoul')
    return df

def plot_to_html(fig, title):
    try:
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        return f"""<div style='width:48%;margin:1%;'>
<h4>{title}</h4>
<img src='data:image/png;base64,{encoded}' style='max-width:100%;height:auto'/>
</div>"""
    except Exception as e:
        return f"<p>{title} 시각화 실패: {e}</p>"

def generate_visuals_for_strategy(strategy_label, strategy_kor):
    html = f"<h2>📊 {strategy_kor} 전략 분석</h2><div style='display:flex;flex-wrap:wrap;'>"

    try:
        df_pred = load_df(PREDICTION_LOG)
    except Exception as e:
        return f"<p>prediction_log.csv 로드 실패: {e}</p></div>"

    try:
        df_audit = load_df(AUDIT_LOG)
    except Exception as e:
        df_audit = pd.DataFrame()
        html += f"<p>audit_log.csv 로드 실패: {e}</p>"

    try:
        df = df_pred[df_pred['strategy'] == strategy_label]
        df['date'] = df['timestamp'].dt.date
        df['result'] = df['status'].map({'success': 1, 'fail': 0})
        sr = df[df['status'].isin(['success', 'fail'])].groupby('date')['result'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5,2))
        ax.plot(sr['date'], sr['result'])
        ax.set_title("📈 최근 성공률 추이")
        html += plot_to_html(fig, "📈 최근 성공률 추이")
    except Exception as e:
        html += f"<p>1번 오류: {e}</p>"

    try:
        df = df_audit[df_audit['strategy'] == strategy_label]
        fig, ax = plt.subplots(figsize=(5,2))
        ax.scatter(df['predicted_return'], df['actual_return'], alpha=0.5)
        ax.set_xlabel("예측 수익률")
        ax.set_ylabel("실제 수익률")
        ax.set_title("🎯 예측 vs 실제 수익률")
        html += plot_to_html(fig, "🎯 예측 vs 실제 수익률")
    except Exception as e:
        html += f"<p>2번 오류: {e}</p>"

    try:
        df = df_audit.dropna(subset=['accuracy_before', 'accuracy_after'])
        df = df[df['strategy'] == strategy_label]
        df['accuracy_before'] = pd.to_numeric(df['accuracy_before'], errors='coerce')
        df['accuracy_after'] = pd.to_numeric(df['accuracy_after'], errors='coerce')
        fig, ax = plt.subplots(figsize=(5,2))
        ax.plot(df['timestamp'], df['accuracy_before'], label="Before")
        ax.plot(df['timestamp'], df['accuracy_after'], label="After")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        ax.set_title("📚 오답학습 전후 정확도 변화")
        html += plot_to_html(fig, "📚 오답학습 전후 정확도 변화")
    except Exception as e:
        html += f"<p>3번 오류: {e}</p>"

    try:
        df = df_pred[df_pred['strategy'] == strategy_label]
        df = df[df['status'].isin(['success', 'fail'])]
        df['result'] = df['status'].map({'success': 1, 'fail': 0})
        df = df.sort_values('timestamp', ascending=False).head(20)
        pivot = df.pivot(index='symbol', columns='timestamp', values='result')
        fig, ax = plt.subplots(figsize=(5,2))
        data = pivot.fillna(0).values if not pivot.empty else np.zeros((1,1))
        ax.imshow(data, cmap='Greens', aspect='auto')
        ax.set_title("🧩 최근 예측 히트맵")
        ax.set_yticks([]); ax.set_xticks([])
        html += plot_to_html(fig, "🧩 최근 예측 히트맵")
    except Exception as e:
        html += f"<p>4번 오류: {e}</p>"

    try:
        df = df_audit[df_audit['strategy'] == strategy_label]
        df = df.dropna(subset=['actual_return']).sort_values('timestamp')
        df['date'] = df['timestamp'].dt.date
        df['cum_return'] = df['actual_return'].cumsum()
        fig, ax = plt.subplots(figsize=(5,2))
        ax.plot(df['date'], df['cum_return'])
        ax.set_title("💰 누적 수익률 추적")
        html += plot_to_html(fig, "💰 누적 수익률 추적")
    except Exception as e:
        html += f"<p>5번 오류: {e}</p>"

    try:
        df = df_pred[df_pred['strategy'] == strategy_label]
        df = df[df['status'].isin(['success', 'fail']) & df['model'].notna()]
        df['result'] = df['status'].map({'success': 1, 'fail': 0})
        df['date'] = df['timestamp'].dt.date
        group = df.groupby(['model', 'date'])['result'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5,2))
        for m in group['model'].unique():
            temp = group[group['model'] == m]
            ax.plot(temp['date'], temp['result'], label=m)
        ax.set_title("🧠 모델별 성공률 변화")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        html += plot_to_html(fig, "🧠 모델별 성공률 변화")
    except Exception as e:
        html += f"<p>6번 오류: {e}</p>"

    try:
        df = df_audit[df_audit['strategy'] == strategy_label]
        df = df.dropna(subset=['predicted_volatility', 'actual_volatility'])
        df['predicted_volatility'] = pd.to_numeric(df['predicted_volatility'], errors='coerce')
        df['actual_volatility'] = pd.to_numeric(df['actual_volatility'], errors='coerce')
        fig, ax = plt.subplots(figsize=(5,2))
        ax.plot(df['timestamp'], df['predicted_volatility'], label="예측 변동성")
        ax.plot(df['timestamp'], df['actual_volatility'], label="실제 변동성")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        ax.set_title("🌪️ 변동성 예측 vs 실제 변동성")
        html += plot_to_html(fig, "🌪️ 변동성 예측 vs 실제 변동성")
    except Exception as e:
        html += f"<p>7번 오류: {e}</p>"

    html += "</div>"
    return html

def generate_visual_report():
    return (
        generate_visuals_for_strategy("단기", "단기") +
        generate_visuals_for_strategy("중기", "중기") +
        generate_visuals_for_strategy("장기", "장기")
    )
