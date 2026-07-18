import sys # 파이썬 실행 환경 관리자
from fastapi import FastAPI # 번역 결과 서빙 계층
from fastapi.responses import StreamingResponse # 실시간 송출
from pydantic import BaseModel
import uvicorn # ASGI 웹 서버
from contextlib import asynccontextmanager

# 비즈니스 로직 -> 번역, 교정
from translation import init_vector_db, init_exact_matches, generate_translation_stream
from spell_correction import init_corrector, correct_text

# ASCII 한글 로그 or 이모지 출력 ignore
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
    yield

app = FastAPI(lifespan=lifespan)

class TranslationRequest(BaseModel): # 요청 데이터 규격
    text: str             # 영어 문장
    draft_text: str = ""  # 구글 1차 번역

@app.post("/translate/stream") # API 엔드 포인트
async def translate_stream(payload: TranslationRequest):
    # 교정된 고유명사, 용어집
    corrected_text, matched_dict = correct_text(payload.text) # 교정
    
    # 교정된 텍스트와 사전을 번역기로 전달
    return StreamingResponse(
        generate_translation_stream(corrected_text, payload.draft_text, matched_dict), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
