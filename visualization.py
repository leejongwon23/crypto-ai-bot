import pandas as pd
import matplotlib.pyplot as plt
import io, base64, numpy as np
from datetime import datetime
import pytz

PREDICTION_LOG = "/persistent/prediction_log.csv"
AUDIT_LOG = "/persistent/logs/evaluation_audit.csv"

def load_df(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
    return df

def plot_to_html(fig, title):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return f"<h4>{title}</h4><img src='data:image/png;base64,{base64.b64encode(buf.read()).decode()}'/><br>"

def generate_visuals_for_strategy(strategy_label, strategy_kor):
    html = f"<h2>📊 {strategy_kor} 전략 분석</h2>"
    try:
        df_pred = load_df(PREDICTION_LOG)
        df_pred = df_pred[df_pred['strategy'] == strategy_label]
        df_pred['date'] = df_pred['timestamp'].dt.date
        df_pred['result'] = df_pred['status'].map({'success':1, 'fail':0})
        sr = df_pred[df_pred['status'].isin(['success','fail'])].groupby('date')['result'].mean().reset_index()
        fig, ax = plt.subplots(); ax.plot(sr['date'], sr['result']); ax.set_title("📈 최근 성공률 추이"); ax.set_ylabel("성공률")
        html += plot_to_html(fig, "📈 최근 성공률 추이")
    except Exception as e: html += f"<p>1번 오류: {e}</p>"

    try:
        df_audit = load_df(AUDIT_LOG)
        df_audit = df_audit[df_audit['strategy'] == strategy_label]
        fig, ax = plt.subplots(); ax.scatter(df_audit['predicted_return'], df_audit['actual_return'], alpha=0.5)
        ax.set_xlabel("예측 수익률"); ax.set_ylabel("실제 수익률"); ax.set_title("🎯 예측 vs 실제 수익률")
        html += plot_to_html(fig, "🎯 예측 수익률 vs 실제 수익률")
    except Exception as e: html += f"<p>2번 오류: {e}</p>"

    try:
        df = df_audit.dropna(subset=['accuracy_before', 'accuracy_after'])
        fig, ax = plt.subplots(); ax.plot(df['timestamp'], df['accuracy_before'], label="Before")
        ax.plot(df['timestamp'], df['accuracy_after'], label="After"); ax.legend(); ax.set_title("📚 오답학습 전후 정확도 변화")
        html += plot_to_html(fig, "📚 오답학습 전후 정확도 변화")
    except Exception as e: html += f"<p>3번 오류: {e}</p>"

    try:
        recent = df_pred[df_pred['status'].isin(['success','fail'])].sort_values('timestamp', ascending=False)
        recent = recent.groupby('strategy').head(20).pivot(index='strategy', columns='timestamp', values='result')
        fig, ax = plt.subplots(figsize=(10, 2)); ax.imshow(recent.fillna(0), cmap='Greens', aspect='auto')
        ax.set_title("🧩 최근 예측 히트맵"); ax.set_yticks([]); ax.set_xticks([])
        html += plot_to_html(fig, "🧩 최근 예측 히트맵")
    except Exception as e: html += f"<p>4번 오류: {e}</p>"

    try:
        df = df_audit.dropna(subset=['actual_return']).sort_values('timestamp')
        df['date'] = df['timestamp'].dt.date
        df['cum_return'] = df.groupby('strategy')['actual_return'].cumsum()
        fig, ax = plt.subplots(); ax.plot(df['date'], df['cum_return']); ax.set_title("💰 누적 수익률 추적")
        html += plot_to_html(fig, "💰 누적 수익률 추적")
    except Exception as e: html += f"<p>5번 오류: {e}</p>"

    try:
        df = df_pred[df_pred['status'].isin(['success','fail']) & df_pred['model'].notna()]
        df['result'] = df['status'].map({'success':1,'fail':0})
        df['date'] = df['timestamp'].dt.date
        group = df.groupby(['model','date'])['result'].mean().reset_index()
        fig, ax = plt.subplots(); 
        for m in group['model'].unique():
            temp = group[group['model']==m]
            ax.plot(temp['date'], temp['result'], label=m)
        ax.set_title("🧠 모델별 성공률 변화"); ax.legend()
        html += plot_to_html(fig, "🧠 모델별 성공률 변화")
    except Exception as e: html += f"<p>6번 오류: {e}</p>"

    try:
        df = df_audit.dropna(subset=['predicted_volatility','actual_volatility'])
        fig, ax = plt.subplots()
        ax.plot(df['timestamp'], df['predicted_volatility'], label="예측 변동성")
        ax.plot(df['timestamp'], df['actual_volatility'], label="실제 변동성")
        ax.set_title("🌪️ 변동성 예측 vs 실제 변동성"); ax.legend()
        html += plot_to_html(fig, "🌪️ 변동성 예측 vs 실제 변동성")
    except Exception as e: html += f"<p>7번 오류: {e}</p>"

    return html

def generate_visual_report():
    return (
        generate_visuals_for_strategy("단기", "단기") +
        generate_visuals_for_strategy("중기", "중기") +
        generate_visuals_for_strategy("장기", "장기")
    )
