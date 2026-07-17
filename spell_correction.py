import os
import re
import spacy
import difflib
import jellyfish
import phonetics
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny
from symspellpy import SymSpell, Verbosity

# 💡 공통 객체 임포트 (중복 생성 방지)
from shared import embed_model, qdrant

nlp = spacy.load("en_core_web_sm") 
COLLECTION_NAME = "league_stt_logical" 

if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE), 
    )

sym_spell = SymSpell(max_dictionary_edit_distance=4, prefix_length=7)
ALL_TERMS = set()
ALL_TERMS_META = {}

IGNORE_TOKENS = { 
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves", "they", "them", "their", "theirs", "themselves",
    "someone", "somebody", "something", "anyone", "anybody", "anything",
    "noone", "nobody", "nothing", "everyone", "everybody", "everything",
    "who", "whom", "whose", "which", "what", "whatever", "whoever", "whichever",
    "some", "any", "every", "all", "both", "neither", "either",
    "each", "much", "many", "few", "fewer", "several", "enough", "such", "other", "another",
    "uh", "um", "ah", "er", "hmm", "like", "just", "actually",
    "basically", "literally", "totally", "seriously", "honestly", "frankly",
    "well", "okay", "ok", "yeah", "yep", "nope", "oh", "wow", "gosh", "jeez", "gee",
    "mhm", "uhhuh", "huh", "phew", "oops", "yay", "yikes", "duh", "blah",
    "fucking", "freaking", "damn", "goddamn", "hell", "shit", "crap", "ass",
    "bitch", "af", "omg", "wtf", "lol", "lmao", "rofl", "bro", "dude", "man", "guys", "yall",
    "bruh", "heck", "lmfao", "noob", "stfu", "pls", "plz",
    "very", "really", "quite", "rather", "somewhat", "too", "also", "even", "only",
    "always", "never", "sometimes", "usually", "often", "already", "still", "yet",
    "almost", "exactly", "probably", "maybe", "perhaps", "certainly", "definitely",
    "absolutely", "simply", "especially", "particularly", "suddenly", "eventually",
    "anyway", "somehow", "anywhere", "everywhere", "nowhere", "somewhere",
    "yes", "no", "true", "false", "right", "wrong", "sure", "please", "thanks", "thank",
    "hello", "hi", "hey", "bye", "goodbye", "morning", "night", "sorry", "excuse",
    "pardon", "welcome", "congrats", "cheers", "mate", "buddy", "pal", "op"
}

def final_rag_chunker(text):
    doc = nlp(text)
    chunks = []
    current_tokens = []
    def save_chunk(): 
        if current_tokens:
            if any(t.pos_ in ["NOUN", "PROPN"] for t in current_tokens):
                chunks.append({"text": " ".join([t.text for t in current_tokens])})
    for token in doc:
        if token.text.lower() in IGNORE_TOKENS: 
            continue 
        if token.pos_ in ["ADJ", "NOUN", "PROPN"]:
            current_tokens.append(token)
        else:
            save_chunk() 
            current_tokens = []
    save_chunk() 
    return chunks

def init_corrector():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "league.txt")
    
    if not os.path.exists(file_path): 
        print(f"⚠️ [교정 모듈]: {file_path} 파일이 없습니다.")
        return False
        
    points = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.startswith("#") or not line.strip(): continue 
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6: continue
            
            term = parts[1].strip().lower()
            ALL_TERMS.add(term)
            
            merged_term = term.replace(" ", "")
            sym_spell.create_dictionary_entry(merged_term, 100) 
            
            p_hash, s_hash = phonetics.dmetaphone(merged_term)
            ALL_TERMS_META[term] = [h for h in (p_hash, s_hash) if h]
            
            vector = embed_model.encode(term).tolist()
            points.append(PointStruct(id=idx, vector=vector, payload={"term": term}))
            
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ [교정 모듈]: 지식베이스 로드 완료.")
    return True

def count_syllables(word):
    return max(1, len(re.findall(r'[aeiouy]+', word.lower())))

def correct_text(stt_text):
    print(f"\n{'='*50}\n[원본 문장] {stt_text}\n{'-'*50}")
    chunks = final_rag_chunker(stt_text)
    modified_text = stt_text
    
    def bi_jaro_winkler(s1, s2):
        if not s1 or not s2: return 0.0
        fwd = jellyfish.jaro_winkler_similarity(s1, s2)
        bwd = jellyfish.jaro_winkler_similarity(s1[::-1], s2[::-1])
        return (fwd + bwd) / 2.0

    for chunk in chunks:
        text = chunk['text'] 
        clean_chunk = re.sub(r'[^\w\s]', '', text).strip().lower()
        if not clean_chunk or clean_chunk in IGNORE_TOKENS: continue
        
        merged_chunk = clean_chunk.replace(" ", "")
        chunk_len = len(merged_chunk)
        chunk_words = clean_chunk.split()
        
        print(f"\n🔍 [분석 타겟]: '{text}' (공백제거: '{merged_chunk}')")
        search_terms = set()
        
        allowed_distance = 1 if chunk_len <= 4 else (2 if chunk_len <= 7 else 3)
        suggestions = sym_spell.lookup(merged_chunk, Verbosity.CLOSEST, max_edit_distance=4)
        sym_candidates = []
        if suggestions:
            for s in suggestions:
                if s.distance <= allowed_distance:
                    matches = [t for t in ALL_TERMS if t.replace(" ", "") == s.term]
                    sym_candidates.extend(matches)
                    for m in matches: search_terms.add(m)
        print(f"   ▶ SymSpell 후보: {list(set(sym_candidates)) if sym_candidates else '없음'}")
                        
        c_hashes_merged = [h for h in phonetics.dmetaphone(merged_chunk) if h]
        pho_candidates = []
        for term in ALL_TERMS:
            term_words = term.split()
            if len(chunk_words) != len(term_words):
                continue
            if not any(t.pos_ in ["NOUN", "PROPN"] for t in nlp(term)):
                continue

            t_clean = term.replace(" ", "")
            if abs(count_syllables(merged_chunk) - count_syllables(t_clean)) > 1:
                continue

            max_char_len = max(len(merged_chunk), len(t_clean))
            if max_char_len > 0 and abs(len(merged_chunk) - len(t_clean)) / max_char_len > 0.3:
                continue

            is_pho_match = False
            if len(chunk_words) > 1:
                word_sims = []
                for cw, tw in zip(chunk_words, term_words):
                    cw_h = [h for h in phonetics.dmetaphone(cw) if h]
                    tw_h = [h for h in phonetics.dmetaphone(tw) if h]
                    if cw_h and tw_h:
                        word_sims.append(max(bi_jaro_winkler(c, t) for c in cw_h for t in tw_h))
                
                if len(word_sims) == len(chunk_words) and all(s >= 0.70 for s in word_sims) and sum(word_sims)/len(word_sims) >= 0.80:
                    is_pho_match = True
            else:
                t_hashes_merged = [h for h in phonetics.dmetaphone(t_clean) if h]
                if c_hashes_merged and t_hashes_merged:
                    if max(bi_jaro_winkler(c, t) for c in c_hashes_merged for t in t_hashes_merged) >= 0.80:
                        is_pho_match = True

            if is_pho_match:
                search_terms.add(term)
                pho_candidates.append(term)
                    
        print(f"   ▶ 발음 매칭 후보: {pho_candidates if pho_candidates else '없음'}")

        if not search_terms: 
            print("   ❌ [기각]: 1, 2단계에서 일치하는 RAG 후보를 전혀 찾지 못함.")
            continue

        v_chunk = embed_model.encode(clean_chunk).tolist()
        res = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=v_chunk,
            query_filter=Filter(must=[FieldCondition(key="term", match=MatchAny(any=list(search_terms)))]),
            limit=max(1, len(search_terms)) 
        ).points

        fetched_terms = {hit.payload['term'] for hit in res}
        candidate_pool = list(fetched_terms.union(set(sym_candidates)))
        candidate_scores = []

        for target in candidate_pool:
            target_clean = target.replace(" ", "")
            target_words = target.split()
            
            if len(chunk_words) != len(target_words):
                continue
            if abs(count_syllables(merged_chunk) - count_syllables(target_clean)) > 1:
                continue
            if merged_chunk == target_clean:
                candidate_scores.append((target, 1.0, "완전일치"))
                continue

            max_len = max(len(merged_chunk), len(target_clean))
            if max_len > 0 and abs(len(merged_chunk) - len(target_clean)) / max_len > 0.4:
                continue
            
            difflib_ratio = difflib.SequenceMatcher(None, merged_chunk, target_clean).ratio()
            jw_sim = bi_jaro_winkler(merged_chunk, target_clean)
            text_score = max(difflib_ratio, jw_sim)
            
            pho_merged = 0
            t_hashes_merged = [h for h in phonetics.dmetaphone(target_clean) if h]
            if c_hashes_merged and t_hashes_merged:
                pho_merged = max(bi_jaro_winkler(c, t) for c in c_hashes_merged for t in t_hashes_merged)
            
            pho_split = 0
            if len(chunk_words) == len(target_words) and len(chunk_words) > 1:
                w_sims = []
                for cw, tw in zip(chunk_words, target_words):
                    cw_h = [h for h in phonetics.dmetaphone(cw) if h]
                    tw_h = [h for h in phonetics.dmetaphone(tw) if h]
                    w_sims.append(max(bi_jaro_winkler(c, t) for c in cw_h for t in tw_h) if cw_h and tw_h else 0)
                if len(w_sims) == len(chunk_words):
                    pho_split = sum(w_sims) / len(w_sims)
                
            pho_score = max(pho_merged, pho_split)
            
            if text_score >= 0.55:
                blended_score = (text_score * 0.4) + (pho_score * 0.6)
                final_score = max(text_score, blended_score)
                detail = f"철자:{text_score:.2f}, 발음:{pho_score:.2f} ➔ 융합:{final_score:.2f}"
            else:
                final_score = text_score
                detail = f"철자미달({text_score:.2f} < 0.55) ➔ 발음점수 차단"
                
            if target in sym_candidates:
                if pho_score >= 0.50:
                    final_score = min(1.0, final_score + 0.30)
                    detail += f" | 🚀 SymSpell 강제 부스트"
                else:
                    detail += f" | ❌ SymSpell 기각(발음 {pho_score:.2f} 미달)"

            candidate_scores.append((target, final_score, detail))

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        print("   ▶ 스코어 보드 (Top 3):")
        for i, (tgt, scr, dtl) in enumerate(candidate_scores[:3]):
            print(f"      {i+1}. '{tgt}' = {scr:.2f} ({dtl})")

        final_threshold = 0.85 if chunk_len <= 5 else 0.80
        
        if candidate_scores:
            best_candidate, best_score, _ = candidate_scores[0]
            if best_score >= final_threshold:
                modified_text = re.sub(rf'\b{re.escape(text)}\b', best_candidate, modified_text, flags=re.IGNORECASE)
                print(f"   🟢 [최종 확정]: '{text}' ➔ '{best_candidate}'")
            else:
                print(f"   ❌ [최종 기각]: 최고점 {best_score:.2f} < 기준점 {final_threshold}")         
    print(f"✨ [결과 문장]: {modified_text}")
    return modified_text
