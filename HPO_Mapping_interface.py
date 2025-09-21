import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# Basic Config
# =========================
API_KEY = "2a0ed6f3-b4b6-4d05-91a3-4285b7fa4a43"
NCBO_URL = "https://data.bioontology.org/annotator"
SEARCH_URL = "https://data.bioontology.org/search"
NLM_URL = "https://clinicaltables.nlm.nih.gov/api/hpo/v3/search"
MONARCH_ANNOTATOR_URL = "https://scigraph-ontology.monarchinitiative.org/scigraph/annotations/annotate"
HP_OBO_PATH = r"C:\Users\13189\Desktop\hp.obo" 

# =========================
# UI Config
# =========================
st.set_page_config(
    page_title="HPO mapping",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
      .confidence-high { color: green; font-weight: bold; }
      .confidence-medium { color: orange; font-weight: bold; }
      .confidence-low { color: red; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

RESULTS_PLACEHOLDER = st.empty()

# =========================
# Global HTTP session + cache
# =========================
def _freeze_kv(d):
    if not d:
        return tuple()
    return tuple(sorted((str(k), str(v)) for k, v in d.items()))

_session = requests.Session()
_adapter = HTTPAdapter(
    max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504]),
    pool_connections=20,
    pool_maxsize=20,
)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
_session.headers.update({"Connection": "keep-alive"})

@st.cache_data(show_spinner=False)
def fetch_json_cached(url, params_frozen, headers_frozen, timeout):
    params = dict(params_frozen)
    headers = dict(headers_frozen)
    r = _session.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_json(url, params=None, headers=None, timeout=30):
    return fetch_json_cached(url, _freeze_kv(params or {}), _freeze_kv(headers or {}), timeout)

# =========================
# Load NER Pipeline
# =========================
MODEL_PATH = r"C:\Users\13189\Desktop\ner_pubmedbert_saved_HPO"

@st.cache_resource
def load_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else -1
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)
    return nlp

ner_model = load_ner_pipeline()

# =========================
# Sentence PubMedBERT for Similarity
# =========================
SENTENCE_EMB_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

@st.cache_resource
def load_sentence_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(SENTENCE_EMB_MODEL_NAME, device=device)
    return model

SENT_MODEL = load_sentence_model()

def _embed_text(text: str) -> np.ndarray:
    if not text:
        return np.zeros((SENT_MODEL.get_sentence_embedding_dimension(),), dtype=np.float32)
    vec = SENT_MODEL.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)

def _embed_texts(texts: list, batch_size: int = 128) -> np.ndarray:
    if not texts:
        return np.empty((0, SENT_MODEL.get_sentence_embedding_dimension()), dtype=np.float32)
    vecs = SENT_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)
    return vecs.astype(np.float32)

def semantic_similarity(a: str, b: str) -> float:
    va = _embed_text(a)
    vb = _embed_text(b)
    if va.shape[0] == 0 or vb.shape[0] == 0:
        return 0.0
    return float(np.dot(va, vb))

def semantic_similarity_max_over_list(query: str, texts: list, cap: int = 15) -> float:
    if not texts:
        return 0.0
    vq = _embed_text(query)
    if vq.shape[0] == 0:
        return 0.0
    sims = []
    for t in texts[:cap]:
        vt = _embed_text(str(t))
        sims.append(0.0 if vt.shape[0] == 0 else float(np.dot(vq, vt)))
    return max(sims) if sims else 0.0

# =========================
# Text utils
# =========================
ADJ_STOPLIST = {
    "mild", "moderate", "severe", "lateral", "acute", "chronic", "progressive",
    "intermittent", "recurrent", "episodic", "generalized", "diffuse", "focal",
    "bilateral", "unilateral", "asymmetric", "symmetric"
}

EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "as", "is", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "am", "are", "not", "no"
}

ADJ_SUFFIXES = {'al', 'ic', 'ous', 'ive', 'able', 'ible', 'ful', 'less', 'ish', 'ian', 'ary', 'ory'}

def is_adjective_word(w: str) -> bool:
    if not w:
        return False
    w = w.strip().lower()
    if w in ADJ_STOPLIST:
        return True
    if len(w) <= 2:
        return False
    return any(w.endswith(suf) for suf in ADJ_SUFFIXES)

def is_single_adjective(text: str) -> bool:
    if not text:
        return False
    tokens = re.findall(r"\b\w+\b", text.lower())
    return len(tokens) == 1 and is_adjective_word(tokens[0])

def clean_term(term):
    return re.sub(r'[^\w\s]', '', term).strip()

def tokenize_lower(s: str):
    return re.findall(r"\b\w+\b", (s or "").lower())

def contains_token_subsequence(haystack: str, needle: str) -> bool:
    H = tokenize_lower(haystack)
    N = tokenize_lower(needle)
    if not H or not N or len(N) > len(H):
        return False
    for i in range(len(H) - len(N) + 1):
        if H[i:i + len(N)] == N:
            return True
    return False

def contains_token_in_order(haystack: str, needle: str) -> bool:
    H = tokenize_lower(haystack)
    N = tokenize_lower(needle)
    if not H or not N:
        return False
    i = 0
    for tok in H:
        if i < len(N) and tok == N[i]:
            i += 1
        if i == len(N):
            return True
    return i == len(N)

# =========================
# Entity Extraction – NER / NCBO / Monarch
# =========================
def extract_entities_with_ner(text):
    with torch.no_grad():
        results = ner_model(text)
    hpo_terms = []
    for entity in results:
        if entity.get('entity_group') == 'HPO_TERM':
            hpo_terms.append(entity.get('word', ''))
    return [t for t in hpo_terms if t]

def extract_entities_with_annotator(text):
    try:
        params = {
            "text": text,
            "ontologies": "HP",
            "longest_only": "true",
            "exclude_numbers": "true",
            "include": "prefLabel,synonym",
        }
        headers = {"Authorization": f"apikey token={API_KEY}"}
        data = fetch_json(NCBO_URL, params=params, headers=headers, timeout=30)

        extracted_terms = []
        for ann in data:
            ac = ann.get("annotatedClass", {}) or {}
            iri = ac.get("@id", "") or ""
            if "HP_" in iri:
                annotations = ann.get("annotations", [])
                if annotations:
                    text_match = annotations[0].get("text", "")
                    if text_match and text_match not in extracted_terms:
                        extracted_terms.append(text_match)
        return extracted_terms
    except Exception as e:
        st.error(f"Error extracting entities with annotator: {e}")
        return []

def _pick_monarch_match_span(item, original_text):
    for k in ("token", "text", "matched", "string", "value"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if isinstance(item.get("obj"), dict):
        lbl = item["obj"].get("label")
        if isinstance(lbl, str) and lbl.strip():
            return lbl.strip()
    anns = item.get("annotations") or item.get("terms") or []
    if isinstance(anns, list) and anns:
        first = anns[0]
        if isinstance(first, dict):
            lbl = first.get("label") or first.get("lbl")
            if isinstance(lbl, str) and lbl.strip():
                return lbl.strip()
    return ""

def extract_entities_with_monarch(text):
    try:
        params = {
            "content": text,
            "include_abbrev": "false",
            "include_synonyms": "true",
            "longest_only": "true",
            "min_length": 3,
            "categories": "phenotype"
        }
        data = fetch_json(MONARCH_ANNOTATOR_URL, params=params, headers=None, timeout=30)

        candidates = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                term = _pick_monarch_match_span(item, text)
                if term:
                    candidates.append(term)
        elif isinstance(data, dict):
            for item in data.get("annotations", []):
                if not isinstance(item, dict):
                    continue
                term = _pick_monarch_match_span(item, text)
                if term:
                    candidates.append(term)

        dedup = []
        for t in candidates:
            if t not in dedup:
                dedup.append(t)
        return dedup
    except Exception:
        return []

# =========================
# Offline hp.obo dictionary: exact + fuzzy
# =========================
def _normalize_phrase(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").lower()).strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_resource
def load_hpo_offline_dict(obo_path: str):
    try:
        if not obo_path or not os.path.exists(obo_path):
            return None, 0
        
        try:
            from pronto import Ontology
        except Exception:
            st.info("pronto module is not installed or failed to import. Skipping offline hp.obo extraction layer.")
            return None, 0

        ont = Ontology(obo_path)
        phrase2ids = {}
        max_tokens = 1

        for term in ont.terms():
            tid = str(term.id)
            if not tid.startswith("HP:"):
                continue
            label = (term.name or "").strip()
            if label:
                n = _normalize_phrase(label)
                if n:
                    phrase2ids.setdefault(n, set()).add((label, tid))
                    max_tokens = max(max_tokens, len(n.split()))

            try:
                for syn in term.synonyms:
                    sdesc = (syn.description or "").strip()
                    if sdesc:
                        n = _normalize_phrase(sdesc)
                        if n:
                            phrase2ids.setdefault(n, set()).add((label, tid))
                            max_tokens = max(max_tokens, len(n.split()))
            except Exception:
                pass

        return phrase2ids, max_tokens
    except Exception as e:
        st.info(f"Failed to load hp.obo, skipping offline extraction layer: {e}")
        return None, 0

PHRASE2IDS, OFFLINE_MAX_TOKENS = load_hpo_offline_dict(HP_OBO_PATH)

@st.cache_resource
def build_offline_phrase_embeddings(phrase2ids):
    """
    返回 (phrases:list[str], embs:np.ndarray[float32(N, D)])
    短语已是归一化文本（_normalize_phrase）
    """
    if not phrase2ids:
        return None, None
    phrases = list(phrase2ids.keys())
    embs = SENT_MODEL.encode(phrases, convert_to_numpy=True, normalize_embeddings=True, batch_size=256).astype(np.float32)
    return phrases, embs

OFFLINE_PHRASES, OFFLINE_EMBS = build_offline_phrase_embeddings(PHRASE2IDS)

# ---- 新增：离线 exact/fuzzy 候选（面向“单条输入 term”的候选生成） ----
def get_offline_exact_candidates(term: str):
    """
    直接在离线 hp.obo 词典上做精确短语匹配（对输入做 _normalize_phrase）。
    返回 [(term, label, hp_id, 0.0, "offline_exact", [])]
    """
    if not PHRASE2IDS:
        return []
    n = _normalize_phrase(term)
    if not n or n not in PHRASE2IDS:
        return []
    out, seen = [], set()
    for label, hp_id in PHRASE2IDS[n]:
        if hp_id in seen:
            continue
        out.append((term, label, hp_id, 0.0, "offline_exact", []))
        seen.add(hp_id)
    return out[:10]

def get_offline_fuzzy_candidates(term: str, top_k: int = 10, threshold: float = 0.78):
    """
    使用 embedding 在 OFFLINE_PHRASES 上找与输入最相似的若干短语，
    再映射到 (label, hp_id)。返回 [(term, label, hp_id, 0.0, "offline_fuzzy", [])]
    """
    if OFFLINE_PHRASES is None or OFFLINE_EMBS is None:
        return []
    q = _normalize_phrase(term)
    if not q:
        return []
    v = _embed_text(q)  # (D,)
    if v.shape[0] == 0:
        return []

    sims = OFFLINE_EMBS @ v  # (N,)
    idx = np.argsort(-sims)  # desc
    out, seen = [], set()
    for i in idx[: max(top_k * 3, top_k)]:  # 给一点冗余，再去重
        if float(sims[i]) < threshold:
            continue
        phrase = OFFLINE_PHRASES[i]
        for label, hp_id in PHRASE2IDS.get(phrase, []):
            if hp_id in seen:
                continue
            out.append((term, label, hp_id, 0.0, "offline_fuzzy", []))
            seen.add(hp_id)
            if len(out) >= top_k:
                break
        if len(out) >= top_k:
            break
    return out

# =========================
# NEW: HP 语义索引（术语名+同义词+定义+所有祖先标签 → tokens）
# =========================
@st.cache_resource
def build_hp_semantic_index(obo_path: str):
    """
    返回 dict[hp_id] -> set(tokens)
    tokens 来自：术语标签、同义词、定义、所有祖先术语的标签。
    若无法加载 pronto 或文件不存在，则返回 {}（保持原有流程）。
    """
    try:
        if not obo_path or not os.path.exists(obo_path):
            return {}
        try:
            from pronto import Ontology
        except Exception:
            st.info("pronto 未安装或导入失败，语义索引禁用（pip install pronto 可启用）。")
            return {}

        ont = Ontology(obo_path)
        idx = {}
        for term in ont.terms():
            tid = str(term.id)
            if not tid.startswith("HP:"):
                continue

            texts = []
            if term.name:
                texts.append(term.name)
            # 同义词
            try:
                texts += [s.description or "" for s in term.synonyms]
            except Exception:
                pass
            # 定义
            try:
                if term.definition:
                    texts.append(str(term.definition))
            except Exception:
                pass
            # 祖先标签
            try:
                for anc in term.superclasses(with_self=True):
                    if getattr(anc, "id", "").startswith("HP:") and getattr(anc, "name", None):
                        texts.append(anc.name)
            except Exception:
                pass

            joined = " ".join(texts)
            toks = set(tokenize_lower(joined))
            toks = {t for t in toks if t not in EN_STOPWORDS and t not in ADJ_STOPLIST and len(t) > 2}
            idx[tid] = toks
        return idx
    except Exception as e:
        st.info(f"构建 hp 语义索引失败，已跳过：{e}")
        return {}

HP_SEM_INDEX = build_hp_semantic_index(HP_OBO_PATH)

def _extract_keywords(s: str):
    toks = tokenize_lower(s)
    return [t for t in toks if t not in EN_STOPWORDS and t not in ADJ_STOPLIST and len(t) > 2]

def _candidate_text(std_term, syns_raw, def_text):
    parts = [std_term or ""]
    if isinstance(syns_raw, dict):
        parts += [str(s) for s in (syns_raw.get("synonyms") or [])]
        if not def_text:
            def_text = syns_raw.get("definition") or ""
    elif isinstance(syns_raw, (list, tuple)):
        parts += [str(s) for s in syns_raw]
    if def_text:
        parts.append(str(def_text))
    return " ".join(parts)

# =========================
# 置信度格式化（<0.005 显示为 0，其他保留两位）
# =========================
def fmt_conf(x) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    v = max(0.0, min(1.0, v))
    if v < 0.005:
        return "0"
    return f"{v:.2f}"

# =========================
# 标准化：远端多源 + 离线 exact/fuzzy 统合
# =========================
def get_annotator_results_from_text(text):
    try:
        params = {
            "text": text,
            "ontologies": "HP",
            "longest_only": "true",
            "exclude_numbers": "true",
            "include": "prefLabel,synonym",
        }
        headers = {"Authorization": f"apikey token={API_KEY}"}
        data = fetch_json(NCBO_URL, params=params, headers=headers, timeout=30)

        results = []
        for ann in data:
            ac = ann.get("annotatedClass", {}) or {}
            iri = ac.get("@id", "") or ""
            label = ac.get("prefLabel") or ac.get("rdfs:label") or ""
            synonyms = ac.get("synonym", []) or []
            if "HP_" in iri:
                hp_id = "HP:" + iri.split("HP_")[-1]
                if not label:
                    try:
                        r2 = fetch_json(NLM_URL, params={"terms": hp_id}, headers=None, timeout=10)
                        if len(r2) >= 4 and r2[1] and r2[3]:
                            label = r2[3][0][0]
                    except Exception:
                        pass
                results.append((label, label, hp_id, 0.9, "annotator_from_text", synonyms))
        return results if results else []
    except Exception as e:
        st.error(f"Error in annotator from text: {e}")
        return []

def get_annotator_results(term):
    try:
        params = {
            "text": term,
            "ontologies": "HP",
            "longest_only": "true",
            "exclude_numbers": "true",
            "include": "prefLabel"
        }
        headers = {"Authorization": f"apikey token={API_KEY}"}
        data = fetch_json(NCBO_URL, params=params, headers=headers, timeout=30)

        results = []
        for ann in data:
            ac = ann.get("annotatedClass", {}) or {}
            iri = ac.get("@id", "") or ""
            label = ann.get("prefLabel") or ac.get("rdfs:label") or ""
            if "HP_" in iri:
                hp_id = "HP:" + iri.split("HP_")[-1]
                if not label:
                    try:
                        r2 = fetch_json(NLM_URL, params={"terms": hp_id}, headers=None, timeout=10)
                        if len(r2) >= 4 and r2[1] and r2[3]:
                            label = r2[3][0][0]
                    except Exception:
                        pass
                results.append((term, label, hp_id, 0.0, "annotator", []))
                break
        return results if results else [(term, term, "Not found", 0, "annotator", [])]
    except Exception as e:
        return [(term, term, f"Error: {e}", 0, "annotator", [])]

def get_roots_only_results(term):
    try:
        params = {"q": term, "ontologies": "HP", "roots_only": "true", "include": "prefLabel"}
        headers = {"Authorization": f"apikey token={API_KEY}"}
        data = fetch_json(SEARCH_URL, params=params, headers=headers, timeout=30)

        results = []
        for item in data.get("collection", [])[:10]:
            pref_label = item.get("prefLabel", "")
            id_val = item.get("@id", "")
            if "HP_" in id_val:
                hp_id = "HP:" + id_val.split("HP_")[-1]
                results.append((term, pref_label, hp_id, 0.0, "roots_only", []))
        return results[:5] if results else [(term, term, "No root nodes found", 0, "roots_only", [])]
    except Exception as e:
        return [(term, term, f"Error: {e}", 0, "roots_only", [])]

def _extract_definition_text(def_field):
    if not def_field:
        return ""
    if isinstance(def_field, list):
        try:
            return " ".join([str(x) for x in def_field if x])
        except Exception:
            return str(def_field[0])
    return str(def_field)

def get_synonym_aware_results(term):
    try:
        params = {
            "q": term,
            "ontologies": "HP",
            "include": "prefLabel,synonym,definition",
            "also_search_properties": "true",
        }
        headers = {"Authorization": f"apikey token={API_KEY}"}
        data = fetch_json(SEARCH_URL, params=params, headers=headers, timeout=30)

        results = []
        for item in data.get("collection", [])[:10]:
            pref_label = item.get("prefLabel", "") or ""
            id_val = item.get("@id", "") or ""
            syns = item.get("synonym", []) or []
            definition_raw = item.get("definition", "")
            def_text = _extract_definition_text(definition_raw)
            if "HP_" in id_val:
                hp_id = "HP:" + id_val.split("HP_")[-1]
                payload = {"synonyms": [str(s) for s in syns], "definition": def_text}
                results.append((term, pref_label, hp_id, 0.0, "synonym_aware", payload))
        return results[:5] if results else [(term, term, "No synonym results found", 0, "synonym_aware", [])]
    except Exception as e:
        return [(term, term, f"Error: {e}", 0, "synonym_aware", [])]

def simple_word_split(term):
    words = term.lower().split()
    nouns, adjectives = [], []
    for word in words:
        if (word in EN_STOPWORDS or word in ADJ_STOPLIST or word.isdigit() or len(word) <= 2):
            continue
        is_adj = any(word.endswith(s) for s in ADJ_SUFFIXES)
        if not is_adj and word in ADJ_STOPLIST:
            is_adj = True
        if is_adj:
            adjectives.append(word)
        else:
            nouns.append(word)
    return nouns, adjectives

def get_combination_results(term):
    nouns, adjectives = simple_word_split(term)
    if not nouns and not adjectives:
        return [(term, term, "No nouns/adjectives found", 0, "combination", [])]

    results = []
    for noun in nouns:
        try:
            params = {"q": noun, "ontologies": "HP", "include": "prefLabel"}
            headers = {"Authorization": f"apikey token={API_KEY}"}
            data = fetch_json(SEARCH_URL, params=params, headers=headers, timeout=30)
            for item in data.get("collection", [])[:3]:
                pref_label = item.get("prefLabel", "")
                id_val = item.get("@id", "")
                if "HP_" in id_val:
                    hp_id = "HP:" + id_val.split("HP_")[-1]
                    results.append((noun, pref_label, hp_id, 0.0, "noun_only", []))
        except Exception:
            continue

    for noun in nouns:
        for adj in adjectives:
            combination = f"{adj} {noun}"
            try:
                params = {"q": combination, "ontologies": "HP", "include": "prefLabel"}
                headers = {"Authorization": f"apikey token={API_KEY}"}
                data = fetch_json(SEARCH_URL, params=params, headers=headers, timeout=30)
                for item in data.get("collection", [])[:3]:
                    pref_label = item.get("prefLabel", "")
                    id_val = item.get("@id", "")
                    if "HP_" in id_val:
                        hp_id = "HP:" + id_val.split("HP_")[-1]
                        results.append((combination, pref_label, hp_id, 0.0, "combination", []))
            except Exception:
                continue

    return results[:5] if results else [(term, term, "No combination results found", 0, "combination", [])]

# =========================
# Rescore + select（新增：通用语义对齐校正，其他规则不变）
# =========================
def _rescore_single(original_term: str, cand_tuple):
    """
    对单个候选进行与主流程一致的重打分，返回 new_conf（0-1）。
    cand_tuple: (term0, std_term, hp_id, confidence, source, syns_raw)
    """
    term0, std_term, hp_id, confidence, source, syns_raw = cand_tuple
    std_term = (std_term or "").strip()

    # 提取同义词/定义
    def_text = None
    if isinstance(syns_raw, dict):
        synonyms = [str(s) for s in syns_raw.get("synonyms", [])]
        def_text = str(syns_raw.get("definition", "") or "").strip() or None
    elif isinstance(syns_raw, (list, tuple)):
        synonyms = [str(s) for s in syns_raw]
    else:
        synonyms = []

    if is_single_adjective(std_term):
        new_conf = 0.0
    else:
        if contains_token_subsequence(original_term, std_term):
            new_conf = 0.95
        else:
            sim_pref = semantic_similarity(original_term, std_term) if std_term else 0.0
            sim_syn = semantic_similarity_max_over_list(original_term, synonyms, cap=15) if synonyms else 0.0
            sim_def = semantic_similarity(original_term, def_text) if def_text else 0.0
            new_conf = max(sim_pref, sim_syn, sim_def)

        # 语义对齐校正（与主流程一致）
        kw_input = set(_extract_keywords(original_term))
        doc_text = _candidate_text(std_term, syns_raw, def_text)
        doc_kw = set(_extract_keywords(doc_text))
        if isinstance(hp_id, str) and hp_id in HP_SEM_INDEX:
            doc_kw |= HP_SEM_INDEX[hp_id]
        coverage = (len(kw_input & doc_kw) / max(1, len(kw_input)))
        long_kw = {w for w in kw_input if len(w) >= 6}
        missing_long = [w for w in long_kw if w not in doc_kw]
        penalty = 0.0
        bonus = 0.0
        if coverage < 0.20:
            penalty += 0.25
        if coverage == 0 and new_conf < 0.75:
            penalty += 0.35
        if missing_long:
            penalty += 0.15
        if contains_token_in_order(original_term, std_term):
            bonus += 0.05
        new_conf = float(max(0.0, min(1.0, new_conf + bonus - penalty)))

    return new_conf

def filter_best_results(all_results):
    best_results = []
    for result_group in all_results:
        original_term = result_group["original_term"]

        all_candidates = []
        # 统一 get，保证即使缺键也安全
        for key in [
            "annotator_results", "roots_only_results", "combination_results",
            "synonym_aware_results", "annotator_from_text_results",
            "offline_exact_results", "offline_fuzzy_results"
        ]:
            all_candidates.extend(result_group.get(key, []))

        valid_candidates = [
            c for c in all_candidates
            if c[2] not in [
                "Not found", "No root nodes found", "No combination results found",
                "No synonym results found"
            ] and not str(c[2]).startswith("Error")
        ]

        if not valid_candidates:
            best_results.append({
                "original_term": original_term,
                "best_matches": [(original_term, "No matches found", "N/A", 0, "none", [])]
            })
            continue

        rescored = []
        for c in valid_candidates:
            term0, std_term, hp_id, confidence, source, syns_raw = c
            new_conf = _rescore_single(original_term, c)
            rescored.append((original_term, (std_term or "").strip(), hp_id, new_conf, source, syns_raw))

        if not rescored:
            best_results.append({"original_term": original_term, "best_matches": [(original_term, "No matches found", "N/A", 0, "none", [])]})
            continue

        rescored.sort(key=lambda x: x[3], reverse=True)

        selected_matches, seen_ids = [], set()
        for cand in rescored:
            if cand[2] in seen_ids:
                continue
            if cand[3] > 0.5:
                selected_matches.append(cand)
                seen_ids.add(cand[2])
            if len(selected_matches) >= 5:
                break

        # 兜底：仍为空时强制返回 top-1
        if not selected_matches:
            selected_matches = [rescored[0]]

        if len(selected_matches) < 3:
            for cand in rescored:
                if cand[2] not in seen_ids and len(selected_matches) < 5:
                    selected_matches.append(cand)
                    seen_ids.add(cand[2])

        best_results.append({"original_term": original_term, "best_matches": selected_matches})

    return best_results

def standardize_hpo(terms, original_text):
    all_results = []
    annotator_from_text_results = get_annotator_results_from_text(original_text)

    for term in terms:
        clean_term_val = clean_term(term)
        words = clean_term_val.split()
        if len(words) == 1 and is_adjective_word(words[0]):
            continue

        annotator_results = get_annotator_results(clean_term_val)
        roots_only_results = get_roots_only_results(clean_term_val)
        combination_results = get_combination_results(clean_term_val)
        synonym_aware_results = get_synonym_aware_results(clean_term_val)

        # 离线 exact / fuzzy 作为候选来源 —— 一定写入键，避免 KeyError
        offline_exact_results = get_offline_exact_candidates(clean_term_val)
        offline_fuzzy_results = get_offline_fuzzy_candidates(clean_term_val)

        term_annotator_from_text = []
        for res in annotator_from_text_results:
            if semantic_similarity(clean_term_val, res[0]) > 0.5:
                term_annotator_from_text.append(res)

        all_results.append({
            "original_term": clean_term_val,
            "annotator_results": annotator_results,
            "roots_only_results": roots_only_results,
            "combination_results": combination_results,
            "synonym_aware_results": synonym_aware_results,
            "annotator_from_text_results": term_annotator_from_text,
            "offline_exact_results": offline_exact_results,
            "offline_fuzzy_results": offline_fuzzy_results
        })

    best_results = filter_best_results(all_results)
    return best_results, all_results

# =========================
# 复制友好：Pipe-Table（ASCII）渲染，带分割线，可直接复制
# =========================
def _to_pipe_table(df: pd.DataFrame) -> str:
    df_str = df.copy()
    for c in df_str.columns:
        df_str[c] = df_str[c].astype(str)

    cols = list(df_str.columns)
    widths = []
    for c in cols:
        max_cell = max([len(c)] + [len(v) for v in df_str[c].tolist()]) if len(df_str) else len(c)
        widths.append(max_cell)

    def fmt_row(values):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(values, widths)) + " |"

    header = fmt_row(cols)
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    rows = [fmt_row(df_str.iloc[i].tolist()) for i in range(len(df_str))]
    table = "\n".join([header, sep] + rows)
    return table

def render_copyable_table(df: pd.DataFrame, header: str | None = None):
    if header:
        st.markdown(f"#### {header}")
    st.markdown(f"```\n{_to_pipe_table(df)}\n```")

# =========================
# 渲染函数：所有结果都通过一个容器渲染（覆盖旧内容）
# =========================
def render_results(best_results, all_results, input_term):
    with RESULTS_PLACEHOLDER.container():
           # ===== Summary（显示最高分；并列最高都显示——以两位小数后的分数来判定并列）=====
        summary_rows = []
        for r in best_results:
            bests = r["best_matches"] or []
            if not bests:
                continue

            # 先把每个候选的分数四舍五入到两位小数，再找最高分
            scored = []
            for m in bests:
                raw = float(m[3] if isinstance(m[3], (int, float)) else 0.0)
                r2 = round(raw, 2)
                scored.append((m, r2))

            max_r2 = max(s for (_, s) in scored)

            # 取所有“按两位小数后 == 最高分”的项作为并列第一
            top_items = [m for (m, s) in scored if s == max_r2]

            # 根据 HPO ID 去重，保持顺序
            seen_ids = set()
            dedup_top = []
            for m in top_items:
                hpo_id = m[2]
                if hpo_id in seen_ids:
                    continue
                seen_ids.add(hpo_id)
                dedup_top.append(m)

            # 兜底：万一空了就保留 top-1
            if not dedup_top:
                dedup_top = [bests[0]]

            for idx, m in enumerate(dedup_top):
                std_term = m[1]
                hpo_id = m[2]
                conf = round(float(m[3] if isinstance(m[3], (int, float)) else 0.0), 2)
                summary_rows.append({
                    "Extracted HPO": input_term if idx == 0 else "",
                    "Best Standard HPO": std_term,
                    "HPO ID": hpo_id,
                    "Confidence": f"{conf:.2f}"
                })

        render_copyable_table(pd.DataFrame(summary_rows), header="Summary")
        st.markdown("---")
        # ===== Per-term results（按分数降序；Pipe-Table）=====
        for result in best_results:
            st.markdown(f"### Term: {result['original_term']}")
            if result['best_matches'][0][2] == "N/A":
                st.info("No good matches found for this term.")
                continue
            rows = []
            for match in result['best_matches']:
                term0, std_term, hpo_id, confidence, source, synonyms = match
                syn_list = []
                if isinstance(synonyms, dict):
                    syn_list = [str(s) for s in (synonyms.get("synonyms") or [])][:3]
                elif isinstance(synonyms, (list, tuple)):
                    syn_list = [str(s) for s in synonyms][:3]
                rows.append({
                    "Standard Term": std_term,
                    "HPO ID": hpo_id,
                    "Confidence": fmt_conf(confidence),
                    "Source": source,
                    "Synonyms (top-3)": ", ".join(syn_list)
                })
            df_rows = pd.DataFrame(rows).sort_values("Confidence", ascending=False, key=lambda s: s.astype(float, errors='ignore'))
            render_copyable_table(df_rows)
            st.markdown("---")

        # ===== Detailed（所有来源重打分；Pipe-Table；允许 0）=====
        with st.expander("Show Detailed Results (All Sources)"):
            for result in all_results:
                st.markdown(f"#### Detailed results for: {result['original_term']}")

                def _build_df(data):
                    if not data:
                        return pd.DataFrame(columns=["Standard Term", "HPO ID", "Confidence"])
                    rows = []
                    for cand in data:
                        term0, std_term, hp_id, confidence, source, syns_raw = cand
                        new_conf = _rescore_single(result["original_term"], cand)
                        rows.append({
                            "Standard Term": (std_term or "").strip(),
                            "HPO ID": hp_id,
                            "Confidence": fmt_conf(new_conf)
                        })
                    return pd.DataFrame(rows).sort_values("Confidence", ascending=False, key=lambda s: s.astype(float, errors='ignore'))

                sections = [
                    ("Annotator Results", result.get('annotator_results', [])),
                    ("Root Nodes Only", result.get('roots_only_results', [])),
                    ("Combinations", result.get('combination_results', [])),
                    ("Synonym Aware", result.get('synonym_aware_results', [])),
                    ("From Text", result.get('annotator_from_text_results', [])),
                    ("Offline Exact", result.get('offline_exact_results', [])),  # 安全 get
                    ("Offline Fuzzy", result.get('offline_fuzzy_results', [])),  # 安全 get
                ]
                for title, data in sections:
                    render_copyable_table(_build_df(data), header=title)

# =========================
# UI（只做标准化：输入即唯一 term）
# =========================
st.markdown("<h1 class='app-title-nowrap'>HPO mapping</h1>", unsafe_allow_html=True)

# 显示离线词典/语义索引状态
if PHRASE2IDS:
    st.caption(
        f"Offline hp.obo loaded: {len(PHRASE2IDS)} phrases; max tokens in phrase = {OFFLINE_MAX_TOKENS}. "
        f"Fuzzy index: {'ready' if OFFLINE_EMBS is not None else 'disabled'}."
    )
else:
    st.caption("Offline hp.obo not loaded (file missing or pronto not installed).")
st.caption(f"HP semantic index: {'enabled' if HP_SEM_INDEX else 'disabled'}.")

# ---- Session state 初始化
if "stored_best_results" not in st.session_state:
    st.session_state.stored_best_results = None
if "stored_all_results" not in st.session_state:
    st.session_state.stored_all_results = None
if "stored_input_term" not in st.session_state:
    st.session_state.stored_input_term = ""

text = st.text_area("Input Text", height=180, placeholder="Input one HPO.....")

colA, colB = st.columns([1, 1])
with colA:
    run_clicked = st.button("Standardize HPO", type="primary")
with colB:
    cleared = st.button("Clear Results")

if cleared:
    # 清空展示与状态（旧结构的缓存一并清掉）
    st.session_state.stored_best_results = None
    st.session_state.stored_all_results = None
    st.session_state.stored_input_term = ""
    RESULTS_PLACEHOLDER.empty()

if run_clicked:
    input_term = (text or "").strip()
    if not input_term:
        st.warning("Please provide some text.")
    else:
        status = st.status("Processing...", expanded=False)
        # 直接把整段输入作为唯一待标准化短语
        best_results, all_results = standardize_hpo([input_term], input_term)
        status.update(label="Standardization completed.", state="complete")

        # 保存到 session_state，并统一渲染（覆盖旧内容）
        st.session_state.stored_best_results = best_results
        st.session_state.stored_all_results = all_results
        st.session_state.stored_input_term = input_term
        render_results(best_results, all_results, input_term)

# 页面刷新或输入变更时，如果上一次结果存在，就显示它（不会残留老的“Term：…”块）
if (not run_clicked) and st.session_state.stored_best_results is not None:
    render_results(
        st.session_state.stored_best_results,
        st.session_state.stored_all_results,
        st.session_state.stored_input_term
    )
