import os
import sys
import re
import asyncio
import glob
import json
import textwrap
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer

# 인코딩 에러 및 강제 종료 차단
if sys.stdin is not None:
    sys.stdin.reconfigure(encoding='utf-8', errors='ignore')
if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

# ---------------------------------------------------------
# 1. 전역 상태 및 문맥 추적기
# ---------------------------------------------------------
CONVO_HISTORY = deque(maxlen=3) # 이전 대화 3턴 기억 (번역 어투 유지용)
LAST_ACTIVE_DOMAIN = "league"   # 현재 대화의 주축 도메인 (기본값 설정)


# 전역 변수 및 캐시 정의
IRREGULAR_VERBS = {}
INVERTED_INDEX = defaultdict(set)
TERM_REGEX_CACHE = {}

LLM_CLIENT = AsyncOpenAI(
    api_key="EMPTY", 
    base_url="http://localhost:8000/v1" # 로컬 SGLang 서버 주소
)

MODEL_NAME = "Qwen/Qwen3-8B-FP8"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")

qdrant = QdrantClient(":memory:")
COLLECTION_NAME = "hybrid_dictionary"

SYSTEM_PROMPT = textwrap.dedent("""\
    # Role
    EN-KR Banmal Translator.
    # Rules
    - Tone: STRICTLY Banmal (반말).
    - Preserve existing KR words.
    - Translate EN parts to natural KR.
    - [CRITICAL] Strictly apply [Glossary] definitions from User Message.
""")


# (기존 전역 변수 아래에 추가)
EXACT_MATCH_DICT = {}

def init_exact_matches():
    global EXACT_MATCH_DICT
    json_path = "/sgl-workspace/sglang/data/quick.json" # 경로 확인 필요
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            EXACT_MATCH_DICT = json.load(f)
        print(f"⚡ [1단계 하드매칭 캐시]: {len(EXACT_MATCH_DICT)}개 로드 완료.")
    else:
        # 테스트용으로 직접 박아넣어도 무방합니다.
        print("⚠️ quick.json 파일이 없어 하드매칭이 비활성화됩니다.")

# --- 비즈니스 로직 함수군 (기존 코드와 동일) ---
def get_verb_forms(verb):
    forms = {verb}
    if verb in IRREGULAR_VERBS:
        forms.update(IRREGULAR_VERBS[verb].split("|"))
        return forms 
    if verb.endswith('e'):
        forms.update([verb + 's', verb[:-1] + 'ed', verb[:-1] + 'ing'])
    else:
        forms.update([
            verb + 's', verb + 'es', verb + 'ed', verb + 'ing',
            verb + verb[-1] + 'ed', verb + verb[-1] + 'ing'
        ])
    return forms

def get_verb_pattern(verb):
    if verb in IRREGULAR_VERBS:
        return rf"(?:{'|'.join(IRREGULAR_VERBS[verb].split('|'))})"
    if verb.endswith('e'):
        return rf"{re.escape(verb[:-1])}(?:e|es|ed|ing)"
    return rf"{re.escape(verb)}(?:s|es|ed|ing|{verb[-1]}ed|{verb[-1]}ing)?"

def build_phrasal_regex(term, pos):
    words = term.split()
    if pos != "verb":
        return rf"\b{re.escape(term)}\b"
    gap = r"(?:\s+\w+){0,4}"
    if len(words) == 1:
        return rf"\b(?:{get_verb_pattern(words[0])})\b"
    first, last = words[0], words[-1]
    return rf"(?:\b(?:{get_verb_pattern(first)}){gap}\s+{re.escape(last)}\b|\b{re.escape(first)}{gap}\s+(?:{get_verb_pattern(last)})\b)"

def init_vector_db():
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embed_model.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        ),
    )
    if os.path.exists("verbs.txt"):
        with open("verbs.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2: 
                    IRREGULAR_VERBS[parts[0].lower()] = "|".join(parts) 

    DATA_DIR = "/sgl-workspace/sglang/data"
    if not os.path.exists(DATA_DIR):
        print(f"❌ 에러: {DATA_DIR} 폴더가 존재하지 않습니다.")
        return
    files_to_load = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    files_to_load = [f for f in files_to_load if "verbs.txt" not in f]
    points = []
    point_id = 0
    loaded_data = []
    
    for file_path in files_to_load:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = [p.strip() for p in line.split("|")]                  
                    if len(parts) >= 6:
                        loaded_data.append((parts[0], parts[1].lower(), parts[2], parts[3], parts[4], parts[5]))
                    elif len(parts) == 5: 
                        loaded_data.append((parts[0], parts[1].lower(), parts[2], parts[3], parts[4], "번역 없음"))
                        
    data_to_embed = loaded_data if loaded_data else []
    texts_to_embed = []
    payloads = []

    for domain, term, pos, meaning, en_ex, kr_ex in data_to_embed:
        if term not in TERM_REGEX_CACHE:
            TERM_REGEX_CACHE[term] = re.compile(build_phrasal_regex(term, pos), re.IGNORECASE)       
            term_words = term.split()
            if pos == "verb":
                for w in term_words:
                    for form in get_verb_forms(w):
                        INVERTED_INDEX[form.lower()].add(term)
            else:
                for w in term_words:
                    INVERTED_INDEX[w.lower()].add(term)
        texts_to_embed.append(f"[{term}] {en_ex}")
        payloads.append({
            "domain": domain, "term": term, "pos": pos, "meaning": meaning, 
            "en_example": en_ex, "kr_example": kr_ex 
        })
        
    if texts_to_embed:
        vectors = embed_model.encode(texts_to_embed, batch_size=32).tolist() 
        for i, (vector, payload) in enumerate(zip(vectors, payloads)):
            points.append(PointStruct(id=point_id + i, vector=vector, payload=payload))
        point_id += len(vectors)
    
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

def retrieve_hybrid(user_input, top_k=4):
    global LAST_ACTIVE_DOMAIN
    
    # 💡 [핵심 패치]: 유저 입력에서 쉼표, 물음표 등을 다 날려버린 순정 텍스트 생성
    clean_input = re.sub(r'[^a-z0-9\s]', '', user_input.lower()).strip()
    
    # 단어(w)를 쪼갤 때 원본(user_input)이 아니라 clean_input에서 쪼갬
    user_words = set(re.findall(r'\b\w+\b', clean_input)) 
    candidate_terms = set()
    for w in user_words:
        if w in INVERTED_INDEX:
            candidate_terms.update(INVERTED_INDEX[w])
            
    raw_present_terms = set()
    for term in candidate_terms:
        # 💡 정규식 검색 대상도 원본(user_input)이 아니라 clean_input으로 검사!!
        if TERM_REGEX_CACHE[term].search(clean_input):
            raw_present_terms.add(term)

    present_terms = set(raw_present_terms)
    for t1 in raw_present_terms:
        for t2 in raw_present_terms:
            if t1 != t2 and t1 in t2:
                if t1 in present_terms:
                    present_terms.remove(t1)

    print(f"✅ [디버그] 정규식 포착 단어: {present_terms}")
    if not present_terms:
        return []

    # 벡터 검색 (유사도 커트라인 없이 전부 가져옴)
    query_vector = embed_model.encode(user_input).tolist()
    term_filter = Filter(must=[FieldCondition(key="term", match=MatchAny(any=list(present_terms)))])
    
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, query_filter=term_filter, limit=100 
    )

    term_groups = defaultdict(list)
    for hit in search_result.points:
        p = hit.payload
        term_groups[p['term']].append({
            "domain": p['domain'], "meaning": p['meaning'], "pos": p['pos'],
            "score": hit.score, "en_example": p.get('en_example', ''), "kr_example": p.get('kr_example', '')
        })

    resolved_matches = []
    current_turn_domains = []

    for term, items in term_groups.items():
        if len(items) == 1:
            # 💡 하드매칭: 뜻이 하나뿐이면 무조건 채택
            best_match = items[0]
            print(f"📌 [단일 뜻 확정]: {term} -> {best_match['meaning']}")
        else:
            # 💡 다의어 해소: 활성화된 도메인(LAST_ACTIVE_DOMAIN) 데이터가 있으면 무조건 우선 채택
            # 없으면 벡터 점수가 가장 높은 것을 채택
            domain_matched_items = [x for x in items if x['domain'] == LAST_ACTIVE_DOMAIN]
            if domain_matched_items:
                best_match = max(domain_matched_items, key=lambda x: x['score'])
                print(f"📌 [도메인 방어 성공]: '{term}' 다의어 중 [{LAST_ACTIVE_DOMAIN}] 도메인 강제 선택")
            else:
                best_match = max(items, key=lambda x: x['score'])
                print(f"📌 [일반 벡터 선택]: '{term}' -> {best_match['domain']} (Score: {best_match['score']:.4f})")
        
        resolved_matches.append({
            "term": term, "domain": best_match["domain"], "pos": best_match["pos"],
            "meaning": best_match["meaning"], "score": best_match["score"]
        })
        current_turn_domains.append(best_match["domain"])

    # 이번 턴에 가장 많이 등장한 도메인으로 LAST_ACTIVE_DOMAIN 업데이트 (문맥 갱신)
    if current_turn_domains:
        LAST_ACTIVE_DOMAIN = max(set(current_turn_domains), key=current_turn_domains.count)
        print(f"🔄 [도메인 갱신]: 현재 활성 도메인 -> {LAST_ACTIVE_DOMAIN}")

    return sorted(resolved_matches, key=lambda x: x['score'], reverse=True)[:top_k]

def get_user_prompt(sentence, resolved_matches):
    base_prompt = f"원문: {sentence.strip()}\n번역:"
    if not resolved_matches: return base_prompt
    dict_lines = [f"- {m['term']}: {m['meaning']}" for m in resolved_matches]
    dict_str = "\n".join(dict_lines)
    return textwrap.dedent(f"""\
        [사전]
        {dict_str}
        {base_prompt}""")

# ---------------------------------------------------------
# 3. FastAPI 서버 엔진
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_vector_db() # ⚠️ 실제 구동 시 주석 해제
    init_exact_matches()
    print("🚀 서버 부팅 완료 및 RAG 딕셔너리 적재 완료.")
    yield

app = FastAPI(lifespan=lifespan)

class TranslationRequest(BaseModel):
    text: str
@app.post("/translate/stream")
async def translate_stream(payload: TranslationRequest):
    user_input = payload.text.strip()
    print(f"\n🗣️ [사용자 발화]: {user_input}")

    # 기존 정규화 코드
    normalized_input = user_input.lower().strip()
    
    # 💡 축약어 강제 전개 (JSON 하드매칭 방어율 상승)
    normalized_input = normalized_input.replace("you're", "you are")
    normalized_input = normalized_input.replace("i'm", "i am")
    normalized_input = normalized_input.replace("it's", "it is")
    normalized_input = normalized_input.replace("that's", "that is")
    normalized_input = normalized_input.replace("don't", "do not")
    
    # 특수기호 제거
    normalized_input = re.sub(r'[^a-z0-9\s]', '', normalized_input).strip()
    
    if normalized_input in EXACT_MATCH_DICT:
        print(f"⚡ [1단계 하드매칭 성공]: LLM 우회 -> {EXACT_MATCH_DICT[normalized_input]}")
        # LLM 안 거치고 즉각 반환 (0ms)
        async def fast_response():
            yield f"data: {EXACT_MATCH_DICT[normalized_input]}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(fast_response(), media_type="text/event-stream")

    # -----------------------------------------------------
    # 🔍 2단계: RAG 정규식 및 유사도 검색 (하드매칭 뚫렸을 때)
    # -----------------------------------------------------
    resolved_matches = retrieve_hybrid(user_input)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in CONVO_HISTORY:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    
    user_prompt_content = get_user_prompt(user_input, resolved_matches)
    messages.append({"role": "user", "content": user_prompt_content})

    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    async def event_generator():
        response_stream = await LLM_CLIENT.completions.create(
            model=MODEL_NAME, prompt=raw_prompt, temperature=0.1, max_tokens=100, 
            extra_body={"enable_thinking": False}, stream=True
        )
        
        full_translation_chunks = []
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].text:
                text_chunk = chunk.choices[0].text
                full_translation_chunks.append(text_chunk)
                yield f"data: {text_chunk}\n\n"
                
        yield "data: [DONE]\n\n"
        
        final_translation = "".join(full_translation_chunks).strip()
        CONVO_HISTORY.append({"user": user_input, "assistant": final_translation})
        print(f"✅ [번역 완료]: {final_translation}\n" + "-"*40)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
