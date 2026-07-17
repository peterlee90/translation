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

class TranslationRequest(BaseModel):
    text: str
    draft_text: str = ""  # 💡 구글 초벌 번역 수신용

@app.post("/translate/stream")
async def translate_stream(payload: TranslationRequest):  # 💡 수정됨
    # 텍스트 교정 및 매칭된 사전 데이터(Dict) 추출
    corrected_text, matched_dict = correct_text(payload.text)
    
    # 교정된 텍스트와 사전을 번역기로 전달
    return StreamingResponse(
        generate_translation_stream(corrected_text, payload.draft_text, matched_dict), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
