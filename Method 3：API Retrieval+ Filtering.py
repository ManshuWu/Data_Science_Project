import re
import json
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple
from functools import lru_cache
from collections import defaultdict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import obonet
from rapidfuzz import process
from rapidfuzz.fuzz import token_set_ratio
import pandas as pd

# =========================
# 配置和常量
# =========================
MODEL_DIR = r"C:\Users\13189\Desktop\ner_pubmedbert_saved_HPO"
MAX_LENGTH = 512
DEVICE = 0 if torch.cuda.is_available() else -1

# Monarch v3 API & ClinicalTables
MONARCH_BASE = "https://api-v3.monarchinitiative.org/v3/api"
CT_HPO_SEARCH = "https://clinicaltables.nlm.nih.gov/api/hpo/v3/search"

FUZZY_CUTOFF = 85
SIM_THRESH = 50

# 页面配置
st.set_page_config(
    page_title="HPO Extraction & Standardization",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================
# Global Styles
# =========================
st.markdown("""
<style>
.stApp [data-testid="stAppViewContainer"] .main .block-container {
    max-width: 900px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
html, body, [class*="css"] {
    font-size: 18px;
    line-height: 1.55;
}
.app-title-nowrap {
    white-space: nowrap;
    overflow-wrap: normal;
    word-break: keep-all;
    font-weight: 700;
    margin: 0 0 0.6rem 0;
}
textarea, .stTextInput input {
    font-size: 16px !important;
}
.stButton > button {
    font-size: 16px;
    padding: 0.5rem 1.25rem;
    border-radius: 8px;
}
table {
    font-size: 16px;
}
thead tr th {
    text-align: center !important;
}
tbody tr td {
    vertical-align: middle;
}
[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown(
    "<h1 class='app-title-nowrap'>HPO Extraction and Standardization</h1>",
    unsafe_allow_html=True
)

# =========================
# 加载资源函数
# =========================
@st.cache_resource
def load_resources():
    """加载NER模型和HPO图谱"""
    # 加载NER模型
    print(">> Loading NER model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=DEVICE
    )
    
    # 加载HPO图谱
    print(">> Loading HPO terms from obo")
    obo_url = "http://purl.obolibrary.org/obo/hp.obo"
    hpo_map = {}
    hpo_id_to_name = {}  # 新增：存储HPO ID到标准名称的映射
    try:
        graph = obonet.read_obo(obo_url)
        for node_id, data in graph.nodes(data=True):
            name = data.get("name")
            if name:
                hpo_map.setdefault(name.lower(), []).append(node_id)
                hpo_id_to_name[node_id] = name  # 存储ID到名称的映射
            for syn in data.get("synonym", []):
                m = re.search(r'"(.+?)"', syn)
                if m:
                    synonym = m.group(1)
                    hpo_map.setdefault(synonym.lower(), []).append(node_id)
    except Exception as e:
        print(f">> Failed to fetch hp.obo: {e}")
    
    hpo_keys = list(hpo_map.keys())
    
    # 创建HTTP会话
    print(">> Creating HTTP session")
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    
    return ner_pipeline, hpo_map, hpo_keys, hpo_id_to_name, s

# 加载资源
ner_pipeline, hpo_map, hpo_keys, hpo_id_to_name, SESSION = load_resources()

# =========================
# 标准化函数
# =========================
@lru_cache(maxsize=10000)
def normalize_via_monarch(text: str) -> Optional[Tuple[str, str]]:
    """使用Monarch API标准化HPO术语，返回(标准术语, HPO ID)"""
    q = text.strip()
    if not q:
        return None
    params = {"q": q, "category": "biolink:PhenotypicFeature", "limit": 5}
    try:
        r = SESSION.get(f"{MONARCH_BASE}/autocomplete", params=params, timeout=6)
        if r.status_code != 200:
            return None
        data = r.json()
        items = data.get("items") or data.get("results") or data
        if not isinstance(items, list):
            return None

        best_curie, best_label, best_score = None, None, -1
        for it in items:
            curie = it.get("id") or it.get("curie")
            label = (it.get("label") or it.get("name") or "").strip()
            cats = it.get("category") or it.get("categories") or []
            if isinstance(cats, str):
                cats = [cats]
            if curie and str(curie).startswith("HP:"):
                if not cats or any("PhenotypicFeature" in c for c in cats):
                    score = token_set_ratio(q.lower(), label.lower()) if label else 0
                    if score > best_score:
                        best_score, best_curie, best_label = score, curie, label
        return (best_label, best_curie) if best_score >= SIM_THRESH else None
    except Exception:
        return None

@lru_cache(maxsize=10000)
def normalize_via_ct(text: str) -> Optional[Tuple[str, str]]:
    """使用ClinicalTables API标准化HPO术语，返回(标准术语, HPO ID)"""
    q = text.strip()
    if not q:
        return None
    try:
        r = SESSION.get(CT_HPO_SEARCH, params={"terms": q, "maxList": 10}, timeout=6)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list) and len(data) >= 2:
            ids, names = data[0], data[1]
            best_id, best_name, best_score = None, None, -1
            for hp_id, name in zip(ids, names):
                if isinstance(hp_id, str) and hp_id.startswith("HP:"):
                    score = token_set_ratio(q.lower(), (name or "").lower())
                    if score > best_score:
                        best_score, best_id, best_name = score, hp_id, name
            return (best_name, best_id) if best_score >= SIM_THRESH else None
    except Exception:
        return None
    return None

def normalize_mention(text: str) -> Tuple[str, str]:
    """标准化HPO术语的主函数，返回(标准术语, HPO ID)"""
    # 1) Monarch v3
    monarch_result = normalize_via_monarch(text)
    if monarch_result:
        return monarch_result
    
    # 2) ClinicalTables
    ct_result = normalize_via_ct(text)
    if ct_result:
        return ct_result
    
    # 3) 本地HPO映射
    key = text.lower().strip()
    if key in hpo_map:
        score = token_set_ratio(key, key)
        if score >= SIM_THRESH:
            hpo_id = hpo_map[key][0]
            # 从映射中获取标准术语名称
            standard_term = hpo_id_to_name.get(hpo_id, text)
            return (standard_term, hpo_id)
    
    # 4) 模糊匹配
    if hpo_keys:
        match = process.extractOne(key, hpo_keys, score_cutoff=FUZZY_CUTOFF)
        if match:
            matched_term = match[0]
            score = token_set_ratio(key, matched_term)
            if score >= SIM_THRESH:
                hpo_id = hpo_map[matched_term][0]
                # 从映射中获取标准术语名称
                standard_term = hpo_id_to_name.get(hpo_id, text)
                return (standard_term, hpo_id)
    
    return (text, "Not found")  # 返回原始文本和"Not found"

# =========================
# 实体提取和标准化函数
# =========================
def extract_and_standardize(text):
    """从文本中提取HPO实体并标准化"""
    # 提取实体
    results = ner_pipeline(text)
    hpo_mentions = []
    
    for ent in results:
        if ent["entity_group"] == "HPO_TERM":
            mention = ent["word"].strip()
            hpo_mentions.append(mention)
    
    # 标准化
    standardized_terms = []
    for mention in hpo_mentions:
        standard_term, hpo_id = normalize_mention(mention)
        standardized_terms.append({
            "mention": mention,
            "standard_term": standard_term,
            "hpo_id": hpo_id
        })
    
    return standardized_terms

# =========================
# UI
# =========================
text = st.text_area(
    "Input Text",
    height=180,
    placeholder="Paste a PubMed sentence or paragraph here..."
)

if st.button("Extract and Standardize HPO"):
    if not text.strip():
        st.warning("Please provide some text.")
    else:
        with st.spinner("Extracting and standardizing HPO terms..."):
            # 提取和标准化
            results = extract_and_standardize(text)
            
            if not results:
                st.info("No HPO terms found.")
            else:
                st.success(f"Found {len(results)} HPO term(s).")
                
                # 创建标签页
                tabs = st.tabs(["Extracted Terms", "Standardized Terms"])
                
                # 提取的术语标签页
                with tabs[0]:
                    st.markdown("#### Extracted HPO Terms")
                    extracted_terms = [r["mention"] for r in results]
                    df_terms = pd.DataFrame({"HPO Term": extracted_terms})
                    st.table(df_terms)
                
                # 标准化的术语标签页
                with tabs[1]:
                    st.markdown("#### Standardized HPO Terms")
                    if results:
                        df_standardized = pd.DataFrame(results)
                        df_standardized = df_standardized.rename(columns={
                            "mention": "Extracted Term",
                            "standard_term": "Standard Term",
                            "hpo_id": "HPO ID"
                        })
                        # 只保留需要的列
                        df_standardized = df_standardized[["Extracted Term", "Standard Term", "HPO ID"]]
                        st.table(df_standardized)
                    else:
                        st.info("No terms to display after standardization.")