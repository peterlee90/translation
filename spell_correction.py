import os
import re
import difflib
import jellyfish
import phonetics
import nltk
from symspellpy import SymSpell, Verbosity
import shared
# NLTK 표준 영단어 사전 및 가벼운 품사 태거 로드
nltk.download('words', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) # 추가
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True) # 💡 핵심 추가
from nltk.corpus import words
ENGLISH_DICT = set(words.words())

sym_spell = SymSpell(max_dictionary_edit_distance=4, prefix_length=7)
ALL_TERMS = set()
TERM_DB = {} # 💡 번역 모듈로 전달할 메타데이터 인메모리 DB

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

def fast_noun_chunker(text):
    tokens = text.split()
    tagged = nltk.pos_tag(tokens)
    chunks = []
    current_chunk = []
    
    for word, pos in tagged:
        # 💡 원본 대소문자는 살리되 특수기호만 제거
        clean_w = re.sub(r'[^\w\s]', '', word)
        clean_lower = clean_w.lower()
        
        if not clean_lower or clean_lower in IGNORE_TOKENS: 
            if current_chunk:
                chunks.append({"text": " ".join(current_chunk)})
                current_chunk = []
            continue 
            
        # NN(명사), JJ(형용사), FW(외래어), NNP(고유명사) 추출
        if pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('FW'):
            current_chunk.append(clean_w) # 💡 특수기호가 제거된 단어 추가
        else:
            if current_chunk:
                chunks.append({"text": " ".join(current_chunk)})
                current_chunk = []
                
    if current_chunk:
        chunks.append({"text": " ".join(current_chunk)})
        
    return chunks

def init_corrector():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "league.txt")
    
    if not os.path.exists(file_path): 
        print(f"⚠️ [교정 모듈]: {file_path} 파일이 없습니다.")
        return False
        
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.startswith("#") or not line.strip(): continue 
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6: continue
            
            term = parts[1].strip().lower()
            ALL_TERMS.add(term)
            
            merged_term = term.replace(" ", "")
            sym_spell.create_dictionary_entry(merged_term, 100) 
            
            # 💡 벡터 삽입 제거, 인메모리 딕셔너리로 저장 (번역기 RAG 직행용)
            TERM_DB[term] = {
                "domain": parts[0], "term": parts[1].strip(), "pos": parts[2], "meaning": parts[3]
            }
            
    print("✅ [교정 모듈]: 초고속 인메모리 DB 로드 완료.")
    return True

def count_syllables(word):
    return max(1, len(re.findall(r'[aeiouy]+', word.lower())))

def correct_text(stt_text):

    # 💡 1단계: 하드 매칭 확인 (사전 검사)
    normalized = re.sub(r'[^a-z0-9\s]', '', stt_text.lower()).strip()
    if normalized in shared.EXACT_MATCH_DICT:
        return stt_text, [] # 교정 로직 타지 않음
    
    print(f"\n{'='*50}\n[원본 문장] {stt_text}\n{'-'*50}")
    modified_text = stt_text
    matched_dict = []
    
    # 💡 [추가] 1. STT가 완벽히 인식한 정답 게임 용어(laning phase 등)를 최우선 스캔
    lower_stt = stt_text.lower()
    for db_info in TERM_DB.values():
        term_original = db_info["term"].lower()
        # 단어 경계(\b)를 포함해 완벽히 일치할 때만 사전에 추가
        if len(term_original) >= 3 and re.search(rf'\b{re.escape(term_original)}\b', lower_stt):
            if db_info not in matched_dict:
                matched_dict.append(db_info)

    # 2. 기존 오타 교정 로직 시작
    chunks = fast_noun_chunker(stt_text)
    
    def bi_jaro_winkler(s1, s2):
        if not s1 or not s2: return 0.0
        fwd = jellyfish.jaro_winkler_similarity(s1, s2)
        bwd = jellyfish.jaro_winkler_similarity(s1[::-1], s2[::-1])
        return (fwd + bwd) / 2.0

    for chunk in chunks:
        text = chunk['text'] 
        clean_chunk = re.sub(r'[^\w\s]', '', text).strip().lower()
        chunk_words_clean = clean_chunk.split()
        # 💡 영어 사전에 있거나 무시 단어면 스킵
        if not clean_chunk or clean_chunk in IGNORE_TOKENS or clean_chunk in ENGLISH_DICT: 
            continue
        
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
        
        # 💡 조기 종료
        if not sym_candidates:
            print("   ❌ [기각]: SymSpell 후보 없음 ➔ 조기 종료")
            continue
                        
        c_hashes_merged = [h for h in phonetics.dmetaphone(merged_chunk) if h]
        pho_candidates = []
        for term in ALL_TERMS:
            term_words = term.split()
            
            t_clean = re.sub(r'[^\w\s]', '', term).lower().replace(" ", "")
            if abs(count_syllables(merged_chunk) - count_syllables(t_clean)) > 1:
                continue

            max_char_len = max(len(merged_chunk), len(t_clean))
            if max_char_len > 0 and abs(len(merged_chunk) - len(t_clean)) / max_char_len > 0.3:
                continue

            is_pho_match = False
            if len(chunk_words) > 1 and len(chunk_words) == len(term_words):
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
            print("   ❌ [기각]: 1, 2단계 매칭 실패")
            continue

        # 💡 [핵심] Qdrant 벡터 연산 완전 제거, 즉시 메모리 비교
        candidate_pool = list(search_terms.union(set(sym_candidates)))
        candidate_scores = []

        for target in candidate_pool:
            target_clean = re.sub(r'[^\w\s]', '', target).lower().replace(" ", "")
            target_words = target.split()
            
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
            if len(chunk_words) > 1 and len(chunk_words) == len(target_words):
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
                
                # 💡 찾은 단어 메타데이터 추가
                if best_candidate in TERM_DB:
                    if TERM_DB[best_candidate] not in matched_dict:
                        matched_dict.append(TERM_DB[best_candidate])
            else:
                print(f"   ❌ [최종 기각]: 최고점 {best_score:.2f} < 기준점 {final_threshold}")         
    print(f"✨ [결과 문장]: {modified_text}")
    return modified_text, matched_dict
