# main_server.py — 신 4.3 설계 기반 FastAPI 서버 엔드포인트

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import traceback
import time

from recommend import recommend_strategy
from telegram_bot import send_recommendation

results = recommend_strategy()
send_recommendation(results)

app = FastAPI()

# ⏳ 쿨타임 전송 제한 (서버 레벨)
last_sent_time = None
cooldown_seconds = 3600  # 1시간

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.get("/run")
def run():
    global last_sent_time
    now = time.time()

    if last_sent_time and now - last_sent_time < cooldown_seconds:
        remaining = int(cooldown_seconds - (now - last_sent_time))
        return JSONResponse(status_code=429, content={
            "status": "cooldown",
            "message": f"⏳ 쿨타임 중입니다. {remaining}초 후 다시 시도하세요."
        })

    try:
        result = recommend_strategy()
        for msg in result:
            send_recommendation(msg)
        last_sent_time = now
        return JSONResponse(content={
            "status": "success",
            "message": "추천 전송 완료",
            "count": len(result)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_server:app", host="0.0.0.0", port=8000, reload=True)
