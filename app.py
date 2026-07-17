import sys
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# 분리된 커스텀 모듈 임포트
from translation import init_vector_db, init_exact_matches, generate_translation_stream
from spell_correction import init_corrector, correct_text

# 인코딩 에러 및 강제 종료 차단
if sys.stdin is not None:
    sys.stdin.reconfigure(encoding='utf-8', errors='ignore')
if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 각 모듈의 DB 및 캐시 초기화
    init_vector_db() 
    init_exact_matches()
    init_corrector()
    print("🚀 서버 부팅 및 모델 적재 완료.")
    yield

app = FastAPI(lifespan=lifespan)

class TextRequest(BaseModel):
    text: str

@app.post("/translate/stream")
async def translate_stream(payload: TextRequest):
    # 1. 텍스트 교정 (전처리)
    corrected_text = correct_text(payload.text)
    
    # 2. 교정된 텍스트로 번역 스트림 생성
    return StreamingResponse(
        generate_translation_stream(corrected_text), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
