import pandas as pd
import requests

def get_kline(symbol="BTCUSDT", interval="1h", limit=500):
    """
    Bybit에서 특정 심볼과 간격(interval)의 캔들 데이터를 가져옵니다.
    """
    url = f"https://api.bybit.com/v2/public/kline/list?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json().get('result', [])
    
    if not data:
        raise ValueError("API 응답에 데이터가 없습니다.")

    df = pd.DataFrame(data)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
    df.set_index('open_time', inplace=True)

    # 열 이름 표준화 (선택)
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })

    df = df.astype({
        'open': 'float',
        'high': 'float',
        'low': 'float',
        'close': 'float',
        'volume': 'float'
    })

    return df

def get_training_data(symbol="BTCUSDT", interval="1h", limit=500):
    """
    모델 학습용 데이터셋을 반환합니다.
    수집된 데이터에 수익률 등 기본 전처리를 적용합니다.
    """
    df = get_kline(symbol=symbol, interval=interval, limit=limit)

    # 간단한 피처 생성 예시 (사용자 정의 가능)
    df['returns'] = df['close'].pct_change()
    df.dropna(inplace=True)

    return df
