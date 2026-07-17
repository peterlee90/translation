import os
import re
import glob
import json
import textwrap
from collections import defaultdict, deque
from transformers import AutoTokenizer
from openai import AsyncOpenAI
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny

# shared.py에서 공통 객체만 임포트
from shared import embed_model, qdrant

CONVO_HISTORY = deque(maxlen=3)
LAST_ACTIVE_DOMAIN = "league"

IRREGULAR_VERBS = {}
INVERTED_INDEX = defaultdict(set)
TERM_REGEX_CACHE = {}
EXACT_MATCH_DICT = {}

LLM_CLIENT = AsyncOpenAI(
    api_key="EMPTY", 
    base_url="http://localhost:8000/v1"
)

MODEL_NAME = "Qwen/Qwen3-8B-FP8"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

def init_exact_matches():
    global EXACT_MATCH_DICT
    json_path = os.path.join(os.path.dirname(__file__), "data", "quick.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            EXACT_MATCH_DICT = json.load(f)
        print(f"⚡ [1단계 하드매칭 캐시]: {len(EXACT_MATCH_DICT)}개 로드 완료.")
    else:
        print("⚠️ quick.json 파일이 없어 하드매칭이 비활성화됩니다.")

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
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    verbs_path = os.path.join(DATA_DIR, "verbs.txt")

    if os.path.exists(verbs_path):
        with open(verbs_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2: 
                    IRREGULAR_VERBS[parts[0].lower()] = "|".join(parts) 

    if not os.path.exists(DATA_DIR):
        print(f"❌ 에러: {DATA_DIR} 폴더가 존재하지 않습니다.")
        return
        
    files_to_load = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    files_to_load = [f for f in files_to_load if os.path.basename(f) != "verbs.txt"]
    
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
    
    clean_input = re.sub(r'[^a-z0-9\s]', '', user_input.lower()).strip()
    user_words = set(re.findall(r'\b\w+\b', clean_input)) 
    candidate_terms = set()
    for w in user_words:
        if w in INVERTED_INDEX:
            candidate_terms.update(INVERTED_INDEX[w])
            
    raw_present_terms = set()
    for term in candidate_terms:
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
            best_match = items[0]
            print(f"📌 [단일 뜻 확정]: {term} -> {best_match['meaning']}")
        else:
            domain_matched_items = [x for x in items if x['domain'] == LAST_ACTIVE_DOMAIN]
            if domain_matched_items:
                best_match = max(domain_matched_items, key=lambda x: x['score'])
                print(f"📌 [도메인 방어 성공]: '{term}' 다의어 중 [{LAST_ACTIVE_DOMAIN}] 도메인 강제 선택")
            else:
                best_match = max(items, key=lambda x: x['score'])
                print(f"📌 [일반 벡터 선택]: '{term}' -> {best_match['domain']} (Score: {best_match['score']:.4f})")

        if best_match["score"] < 0.7:
            print(f"⚠️ [유사도 미달]: '{term}' 무시 (Score: {best_match['score']:.4f})")
            continue
        
        resolved_matches.append({
            "term": term, "domain": best_match["domain"], "pos": best_match["pos"],
            "meaning": best_match["meaning"], "score": best_match["score"]
        })
        current_turn_domains.append(best_match["domain"])

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

async def generate_translation_stream(user_input: str, matched_dict: list):
    print(f"\n🗣️ [번역기 입력]: {user_input}")
    yield f"data: [CORRECTED]{user_input}\n\n"

    normalized_input = user_input.lower().strip()
    normalized_input = normalized_input.replace("you're", "you are")
    normalized_input = normalized_input.replace("i'm", "i am")
    normalized_input = normalized_input.replace("it's", "it is")
    normalized_input = normalized_input.replace("that's", "that is")
    normalized_input = normalized_input.replace("don't", "do not")
    normalized_input = re.sub(r'[^a-z0-9\s]', '', normalized_input).strip()
    
    if normalized_input in EXACT_MATCH_DICT:
        print(f"⚡ [1단계 하드매칭 성공]: LLM 우회 -> {EXACT_MATCH_DICT[normalized_input]}")
        yield f"data: {EXACT_MATCH_DICT[normalized_input]}\n\n"
        yield "data: [DONE]\n\n"
        return

    resolved_matches = matched_dict
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in CONVO_HISTORY:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    
    user_prompt_content = get_user_prompt(user_input, resolved_matches)
    messages.append({"role": "user", "content": user_prompt_content})

    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

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
