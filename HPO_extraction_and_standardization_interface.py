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
    page_title="HPO Extraction and Standardization",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.result-box { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9; }
.result-title { font-weight: bold; margin-bottom: 10px; color: #333; }
.confidence-high { color: green; font-weight: bold; }
.confidence-medium { color: orange; font-weight: bold; }
.confidence-low { color: red; font-weight: bold; }
.synonym-list { font-size: 14px; color: #666; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

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
    nlp = pipeline("ner", model=model, tokenizer=tokenizer,
                   aggregation_strategy="simple", device=device)
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
    va = _embed_text(a); vb = _embed_text(b)
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
    "mild", "moderate", "severe", "lateral", "acute", "chronic",
    "progressive", "intermittent", "recurrent", "episodic", "generalized",
    "diffuse", "focal", "bilateral", "unilateral", "asymmetric", "symmetric"
}
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "as", "is", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "it", "its", "they", "them",
    "their", "this", "that", "these", "those", "am", "are", "not", "no"
}
ADJ_SUFFIXES = {'al', 'ic', 'ous', 'ive', 'able', 'ible', 'ful', 'less', 'ish', 'ian', 'ary', 'ory'}

def is_adjective_word(w: str) -> bool:
    if not w: return False
    w = w.strip().lower()
    if w in ADJ_STOPLIST: return True
    if len(w) <= 2: return False
    return any(w.endswith(suf) for suf in ADJ_SUFFIXES)

def is_single_adjective(text: str) -> bool:
    if not text: return False
    tokens = re.findall(r"\b\w+\b", text.lower())
    return len(tokens) == 1 and is_adjective_word(tokens[0])

def clean_term(term):
    return re.sub(r'[^\w\s]', '', term).strip()

def tokenize_lower(s: str):
    return re.findall(r"\b\w+\b", (s or "").lower())

def contains_token_subsequence(haystack: str, needle: str) -> bool:
    H = tokenize_lower(haystack); N = tokenize_lower(needle)
    if not H or not N or len(N) > len(H): return False
    for i in range(len(H) - len(N) + 1):
        if H[i:i+len(N)] == N: return True
    return False

def contains_token_in_order(haystack: str, needle: str) -> bool:
    H = tokenize_lower(haystack); N = tokenize_lower(needle)
    if not H or not N: return False
    i = 0
    for tok in H:
        if i < len(N) and tok == N[i]:
            i += 1
            if i == len(N): return True
    return i == len(N)

# =========================
# Entity Extraction
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
            "text": text, "ontologies": "HP",
            "longest_only": "true", "exclude_numbers": "true",
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
                if not isinstance(item, dict): continue
                term = _pick_monarch_match_span(item, text)
                if term: candidates.append(term)
        elif isinstance(data, dict):
            for item in data.get("annotations", []):
                if not isinstance(item, dict): continue
                term = _pick_monarch_match_span(item, text)
                if term: candidates.append(term)
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
    if not phrase2ids:
        return None, None
    phrases = list(phrase2ids.keys())
    embs = SENT_MODEL.encode(phrases, convert_to_numpy=True, normalize_embeddings=True, batch_size=256).astype(np.float32)
    return phrases, embs

OFFLINE_PHRASES, OFFLINE_EMBS = build_offline_phrase_embeddings(PHRASE2IDS)

def extract_entities_with_offline_dict(text: str):
    if not PHRASE2IDS or OFFLINE_MAX_TOKENS <= 0 or not text:
        return []
    raw_tokens = re.findall(r"\w+|-", text)
    lower_tokens = [t.lower() for t in raw_tokens]
    n = len(lower_tokens)
    matched = [False] * n
    hits = []

    for win in range(OFFLINE_MAX_TOKENS, 0, -1):
        i = 0
        while i + win <= n:
            if any(matched[i:i+win]):
                i += 1
                continue
            phrase_norm = _normalize_phrase(" ".join(lower_tokens[i:i+win]))
            if phrase_norm in PHRASE2IDS:
                span_text = " ".join(raw_tokens[i:i+win])
                if span_text not in hits:
                    hits.append(span_text)
                for k in range(i, i+win):
                    matched[k] = True
                i += win
            else:
                i += 1
    return hits

def extract_entities_with_offline_fuzzy(text: str,
                                        FUZZY_MAX_WINDOW: int = 6,
                                        FUZZY_SIM_THRESHOLD: float = 0.78):
    """
    Fuzzy extraction of hp.obo dictionary by using Sentence-BERT (batch calculation, constant threshold/logic)
    """
    if OFFLINE_PHRASES is None or OFFLINE_EMBS is None or not text:
        return []

    raw_tokens = re.findall(r"\w+|-", text)
    lower_tokens = [t.lower() for t in raw_tokens]
    n = len(lower_tokens)
    FUZZY_MAX_WINDOW = max(1, min(FUZZY_MAX_WINDOW, n))

    spans = []  # [(i, win, raw_text, span_lower, span_norm)]
    for win in range(FUZZY_MAX_WINDOW, 0, -1):
        for i in range(0, n - win + 1):
            span_lower = " ".join(lower_tokens[i:i+win]).strip()
            if len(span_lower) < 3:
                continue
            span_norm = _normalize_phrase(span_lower)
            if PHRASE2IDS and span_norm in PHRASE2IDS:
                continue
            span_text = " ".join(raw_tokens[i:i+win]).strip()
            spans.append((i, win, span_text, span_lower, span_norm))

    if not spans:
        return []

    span_lowers = [s[3] for s in spans]
    span_embs = _embed_texts(span_lowers, batch_size=128)  # (M, D)

    # Multiply the block matrix by: (N, D) @ (D, M) -> (N, M) to get the maximum similarity.
    M = span_embs.shape[0]
    block = 256
    sims_max = np.empty((M,), dtype=np.float32)
    for s in range(0, M, block):
        e = min(M, s + block)
        sim_block = OFFLINE_EMBS @ span_embs[s:e].T
        sims_max[s:e] = sim_block.max(axis=0).astype(np.float32)

    hits, seen = [], set()
    for idx, simv in enumerate(sims_max.tolist()):
        if simv >= FUZZY_SIM_THRESHOLD:
            span_text = spans[idx][2]
            if span_text not in seen:
                hits.append(span_text)
                seen.add(span_text)
    return hits

# =========================
# Combined extraction
# =========================
def extract_entities(text):
    annotator_terms = extract_entities_with_annotator(text)
    monarch_terms   = extract_entities_with_monarch(text)
    offline_exact   = extract_entities_with_offline_dict(text)
    offline_fuzzy   = extract_entities_with_offline_fuzzy(text)  # NEW
    ner_terms       = extract_entities_with_ner(text)

    final_terms = []
    anchor_pool = annotator_terms + monarch_terms + offline_exact + offline_fuzzy

    # Extraction mainly depends on NER model, and other source methods are only supplements, because they are more willing to 
    # believe in the ability of NER model. If all other methods fail, we use the results of all NER as the "bottom" because it 
    # may be the only way to recognize certain terms. The treatment rules of NER terms are more relaxed, which allows terms to be 
    # added, even if the similarity with existing terms is low.
    for ner_term in ner_terms:
        if (anchor_pool and any(semantic_similarity(ner_term, a) > 0.6 for a in anchor_pool)) or (not anchor_pool):
            if ner_term not in final_terms:
                final_terms.append(ner_term)

    # Adopt strict de-duplication strategy for the results from other sources. Only when a new term is significantly different
    # from all terms in the merged list (similarity ≤0.6) will it be added. It prevents the same concept from different APIs 
    # from being added repeatedly in a slightly different form.
    for t in annotator_terms:
        if not any(semantic_similarity(t, x) > 0.6 for x in final_terms):
            final_terms.append(t)

    for t in monarch_terms:
        if not any(semantic_similarity(t, x) > 0.6 for x in final_terms):
            final_terms.append(t)

    for t in offline_exact:
        if not any(semantic_similarity(t, x) > 0.6 for x in final_terms):
            final_terms.append(t)

    for t in offline_fuzzy:
        if not any(semantic_similarity(t, x) > 0.6 for x in final_terms):
            final_terms.append(t)

    return final_terms

# =========================
# Multi-source standardization
# =========================
def get_annotator_results_from_text(text):
    try:
        params = {
            "text": text, "ontologies": "HP",
            "longest_only": "true", "exclude_numbers": "true",
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
            "text": term, "ontologies": "HP",
            "longest_only": "true", "exclude_numbers": "true",
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
    if not def_field: return ""
    if isinstance(def_field, list):
        try:
            return " ".join([str(x) for x in def_field if x])
        except Exception:
            return str(def_field[0])
    return str(def_field)

def get_synonym_aware_results(term):
    try:
        params = {
            "q": term, "ontologies": "HP",
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
        if not is_adj and word in ADJ_STOPLIST: is_adj = True
        if is_adj: adjectives.append(word)
        else: nouns.append(word)
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
# Rescore + select
# =========================
def filter_best_results(all_results):
    best_results = []
    for result_group in all_results:
        original_term = result_group["original_term"]

        all_candidates = []
        for key in ["annotator_results", "roots_only_results", "combination_results",
                    "synonym_aware_results", "annotator_from_text_results"]:
            if key in result_group:
                all_candidates.extend(result_group[key])

        valid_candidates = [
            c for c in all_candidates
            if c[2] not in ["Not found", "No root nodes found", "No combination results found", "No synonym results found"]
            and not str(c[2]).startswith("Error")
        ]

        if not valid_candidates:
            best_results.append({
                "original_term": original_term,
                "best_matches": [(original_term, "No matches found", "N/A", 0, "none", [])]
            })
            continue

        rescored = []
        for c in valid_candidates:
            std_term = (c[1] or "").strip()
            hp_id    = c[2]
            source   = c[4]
            syns_raw = c[5] if len(c) > 5 else []

            def_text = None
            if isinstance(syns_raw, dict):
                synonyms = [str(s) for s in syns_raw.get("synonyms", [])]
                def_text = str(syns_raw.get("definition", "") or "").strip() or None
            elif isinstance(syns_raw, (list, tuple)):
                synonyms = [str(s) for s in syns_raw]
            else:
                synonyms = []

            if is_single_adjective(std_term):
                continue

            if contains_token_subsequence(original_term, std_term):
                new_conf = 0.95
            else:
                sim_pref = semantic_similarity(original_term, std_term) if std_term else 0.0
                sim_syn  = semantic_similarity_max_over_list(original_term, synonyms, cap=15) if synonyms else 0.0
                sim_def  = semantic_similarity(original_term, def_text) if def_text else 0.0
                new_conf = max(sim_pref, sim_syn, sim_def)

            rescored.append((original_term, std_term, hp_id, new_conf, source, syns_raw))

        if not rescored:
            best_results.append({"original_term": original_term,
                                 "best_matches": [(original_term, "No matches found", "N/A", 0, "none", [])]})
            continue

        rescored.sort(key=lambda x: x[3], reverse=True)

        selected_matches, seen_ids = [], set()
        for cand in rescored:
            if cand[2] in seen_ids: continue
            if cand[3] > 0.5:
                selected_matches.append(cand)
                seen_ids.add(cand[2])
            if len(selected_matches) >= 5: break

        # The fallback: if still empty, forcefully return the top-1.
        if not selected_matches:
            selected_matches = [rescored[0]]

        if len(selected_matches) < 3:
            for cand in rescored:
                if cand[2] not in seen_ids and len(selected_matches) < 5:
                    selected_matches.append(cand)
                    seen_ids.add(cand[2])

        best_results.append({"original_term": original_term,
                             "best_matches": selected_matches})
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
            "annotator_from_text_results": term_annotator_from_text
        })

    best_results = filter_best_results(all_results)
    return best_results, all_results

# =========================
# UI
# =========================
st.markdown("<h1 class='app-title-nowrap'>HPO Extraction and Standardization</h1>", unsafe_allow_html=True)

# Display offline dictionary loading status
if PHRASE2IDS:
    st.caption(f"Offline hp.obo loaded: {len(PHRASE2IDS)} phrases; max tokens in phrase = {OFFLINE_MAX_TOKENS}. "
               f"Fuzzy index: {'ready' if OFFLINE_EMBS is not None else 'disabled'}.")
else:
    st.caption("Offline hp.obo not loaded (file missing or pronto not installed).")

text = st.text_area("Input Text", height=180, placeholder="Paste a PubMed sentence or paragraph here...")

if st.button("Extract and Standardize HPO"):
    if not text.strip():
        st.warning("Please provide some text.")
    else:
        status = st.empty()
        status.write("Processing...")

        extracted_terms = extract_entities(text)

        filtered_extracted_terms = []
        for t in extracted_terms:
            t_clean = clean_term(t)
            if not is_single_adjective(t_clean):
                filtered_extracted_terms.append(t)

        if not filtered_extracted_terms:
            status.empty()
            st.info("No HPO terms found.")
        else:
            st.success(f"Found {len(filtered_extracted_terms)} HPO term(s).")

            best_results, all_results = standardize_hpo(filtered_extracted_terms, text)

            # Summary（ Allow interval matching to cover e.g. bilateral progressive ptosis -> bilateral ptosis）
            summary_rows = []
            for r in best_results:
                orig = r["original_term"]
                bests = r["best_matches"] or []
                covered = [m for m in bests if contains_token_in_order(orig, m[1])]
                if not covered and bests:
                    covered = [bests[0]]
                covered.sort(key=lambda m: (m[3] if isinstance(m[3], (int, float)) else 0.0), reverse=True)
                for idx, m in enumerate(covered):
                    std_term = m[1]; hpo_id = m[2]
                    conf = m[3] if isinstance(m[3], (int, float)) else 0.0
                    summary_rows.append({
                        "Extracted HPO": orig if idx == 0 else "",
                        "Best Standard HPO": std_term,
                        "HPO ID": hpo_id,
                        "Confidence": f"{conf:.2f}"
                    })
            st.markdown("#### Summary")
            st.markdown(pd.DataFrame(summary_rows).to_html(index=False), unsafe_allow_html=True)
            st.markdown("---")

            # Show one by one
            for result in best_results:
                st.markdown(f"### Term: {result['original_term']}")
                if result['best_matches'][0][2] == "N/A":
                    st.info("No good matches found for this term.")
                    continue

                matches_data = []
                for match in result['best_matches']:
                    term, std_term, hpo_id, confidence, source, synonyms = match
                    if confidence > 0.8:
                        conf_class = "confidence-high"; conf_text = "High"
                    elif confidence > 0.5:
                        conf_class = "confidence-medium"; conf_text = "Medium"
                    else:
                        conf_class = "confidence-low"; conf_text = "Low"

                    # Synonym display (supporting dict structure)
                    synonyms_html = ""
                    try:
                        syn_list = []
                        if isinstance(synonyms, dict):
                            syn_list = [str(s) for s in (synonyms.get("synonyms") or [])][:3]
                        elif isinstance(synonyms, (list, tuple)):
                            syn_list = [str(s) for s in synonyms][:3]
                        if syn_list:
                            synonyms_html = f'<div class="synonym-list">Synonyms: {", ".join(syn_list)}</div>'
                    except Exception:
                        synonyms_html = ""

                    matches_data.append({
                        "Standard Term": f"{std_term}{synonyms_html}",
                        "HPO ID": hpo_id,
                        "Confidence": f'<span class="{conf_class}">{conf_text} ({confidence:.2f})</span>',
                        "Source": source
                    })

                df = pd.DataFrame(matches_data)
                st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                st.markdown("---")

            # Detailed results
            with st.expander("Show Detailed Results (All Sources)"):
                for result in all_results:
                    st.markdown(f"#### Detailed results for: {result['original_term']}")
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown('<div class="result-title">Annotator Results</div>', unsafe_allow_html=True)
                        df_annotator = pd.DataFrame(
                            result['annotator_results'],
                            columns=["Term", "Standard Term", "HPO ID", "Confidence", "Source", "Synonyms"]
                        )
                        st.table(df_annotator[["Standard Term", "HPO ID", "Confidence"]])
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown('<div class="result-title">Root Nodes Only</div>', unsafe_allow_html=True)
                        df_roots = pd.DataFrame(
                            result['roots_only_results'],
                            columns=["Term", "Standard Term", "HPO ID", "Confidence", "Source", "Synonyms"]
                        )
                        st.table(df_roots[["Standard Term", "HPO ID", "Confidence"]])
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col3:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown('<div class="result-title">Combinations</div>', unsafe_allow_html=True)
                        df_combinations = pd.DataFrame(
                            result['combination_results'],
                            columns=["Term", "Standard Term", "HPO ID", "Confidence", "Source", "Synonyms"]
                        )
                        st.table(df_combinations[["Standard Term", "HPO ID", "Confidence"]])
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col4:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown('<div class="result-title">Synonym Aware</div>', unsafe_allow_html=True)
                        df_synonyms = pd.DataFrame(
                            result['synonym_aware_results'],
                            columns=["Term", "Standard Term", "HPO ID", "Confidence", "Source", "Synonyms"]
                        )
                        st.table(df_synonyms[["Standard Term", "HPO ID", "Confidence"]])
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col5:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown('<div class="result-title">From Text</div>', unsafe_allow_html=True)
                        df_from_text = pd.DataFrame(
                            result['annotator_from_text_results'],
                            columns=["Term", "Standard Term", "HPO ID", "Confidence", "Source", "Synonyms"]
                        )
                        st.table(df_from_text[["Standard Term", "HPO ID", "Confidence"]])
                        st.markdown('</div>', unsafe_allow_html=True)

            status.empty()