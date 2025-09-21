# hpo_metrics.py
# -*- coding: utf-8 -*-
import argparse
import re
import pandas as pd
from collections import defaultdict

def norm_id(x):
    """Normalize to HP:XXXXXXX; supports HP_, HP:, HP (space); grabs digits and zero-pads to 7."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    m = re.search(r"(?i)HP[:_ ]?(\d+)", s)
    if not m:
        return None
    return f"HP:{int(m.group(1)):07d}"

def is_nonempty_text(x):
    if x is None:
        return False
    s = str(x)
    return s.strip() != "" and s.lower() not in ["nan", "none", ""]

def norm_term(s):
    """Lenient: lowercase, remove punctuation (keep letters/digits/space/_), collapse spaces."""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)     # remove punctuation
    s = re.sub(r"\s+", " ", s).strip() # collapse spaces
    return s

def parse_obo_file(obo_path):
    """Parse the hp.obo file to get parent-child relationships and term names."""
    term_parents = defaultdict(set)
    term_names = {}
    current_id = None
    current_name = None
    
    with open(obo_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == "[Term]":
                # Save previous term
                if current_id and current_name:
                    term_names[current_id] = current_name
                current_id = None
                current_name = None
            elif line.startswith("id: "):
                current_id = line[4:].strip()
            elif line.startswith("name: "):
                current_name = line[6:].strip()
            elif line.startswith("is_a: "):
                if current_id:
                    parent_id = line.split(" ")[1].strip()
                    term_parents[current_id].add(parent_id)
        
        # Save the last term
        if current_id and current_name:
            term_names[current_id] = current_name
    
    return term_parents, term_names

def build_ancestor_cache(term_parents):
    """Build a cache of all ancestors for each term."""
    ancestor_cache = {}
    
    def get_ancestors(term_id):
        if term_id in ancestor_cache:
            return ancestor_cache[term_id]
        
        ancestors = set()
        for parent in term_parents.get(term_id, set()):
            ancestors.add(parent)
            ancestors.update(get_ancestors(parent))
        
        ancestor_cache[term_id] = ancestors
        return ancestors
    
    # Build cache for all terms
    for term_id in list(term_parents.keys()):
        get_ancestors(term_id)
    
    return ancestor_cache

def is_ancestor(pred_id, gold_id, ancestor_cache):
    """Check if predicted ID is an ancestor of the gold ID."""
    if pred_id == gold_id:
        return True
    
    if gold_id in ancestor_cache:
        return pred_id in ancestor_cache[gold_id]
    
    return False

def main():
    ap = argparse.ArgumentParser(description="Compute HPO standardization metrics from Excel.")
    ap.add_argument("--in", dest="in_path", required=False,
                    default=r"C:\Users\13189\Desktop\61.25%.xlsx",
                    help="Excel file path")
    ap.add_argument("--obo-file", dest="obo_file_path", required=False,
                    default=r"C:\Users\13189\Desktop\hp.obo",
                    help="Path to HPO OBO file")
    ap.add_argument("--pred-term-col", default="Span Standardized Term (code)")
    ap.add_argument("--pred-id-col",   default="Span Standardized HPO ID (code)")
    ap.add_argument("--gold-term-col", default="Final HPO Term")
    ap.add_argument("--gold-id-col",   default="Final HPO ID")
    args = ap.parse_args()

    # Load HPO ontology
    print("Loading HPO ontology...")
    term_parents, term_names = parse_obo_file(args.obo_file_path)
    ancestor_cache = build_ancestor_cache(term_parents)
    print(f"Loaded {len(term_parents)} terms with parent relationships")
    
    # Load Excel data
    df = pd.read_excel(args.in_path, engine="openpyxl")

    # Check columns
    required_cols = [args.pred_term_col, args.pred_id_col, args.gold_term_col, args.gold_id_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("Column(s) not found:", missing)
        print("Available columns:", list(df.columns))
        return

    pred_term = df[args.pred_term_col]
    pred_id   = df[args.pred_id_col]
    gold_term = df[args.gold_term_col]
    gold_id   = df[args.gold_id_col]

    # Coverage
    total_rows = len(df)
    term_cov_n = int(pred_term.apply(is_nonempty_text).sum())
    id_cov_n   = int(pred_id.apply(norm_id).notna().sum())
    term_cov = 100.0 * term_cov_n / total_rows if total_rows else 0.0
    id_cov   = 100.0 * id_cov_n   / total_rows if total_rows else 0.0

    # Normalize IDs
    pred_id_norm = pred_id.apply(norm_id)
    gold_id_norm = gold_id.apply(norm_id)
    
    # ID-based accuracy (exact and ancestor)
    id_mask = pred_id_norm.notna() & gold_id_norm.notna()
    id_den  = int(id_mask.sum())
    
    # Exact ID matches
    id_num_exact = 0
    # Ancestor-based ID matches
    id_num_ancestor = 0
    
    for i in range(len(df)):
        if id_mask.iloc[i]:
            pred = pred_id_norm.iloc[i]
            gold = gold_id_norm.iloc[i]
            
            # Exact match
            if pred == gold:
                id_num_exact += 1
                id_num_ancestor += 1
            # Ancestor match
            elif is_ancestor(pred, gold, ancestor_cache):
                id_num_ancestor += 1
    
    id_acc_exact = 100.0 * id_num_exact / id_den if id_den else 0.0
    id_acc_ancestor = 100.0 * id_num_ancestor / id_den if id_den else 0.0

    # Term-based accuracy (exact and ancestor)
    term_mask = pred_term.apply(is_nonempty_text) & gold_term.apply(is_nonempty_text)
    term_den = int(term_mask.sum())
    
    # Exact term matches
    term_num_exact = 0
    # Ancestor-based term matches
    term_num_ancestor = 0
    
    for i in range(len(df)):
        if term_mask.iloc[i]:
            pred = pred_term.iloc[i]
            gold = gold_term.iloc[i]
            gold_id_val = gold_id_norm.iloc[i]
            
            # Exact match
            if norm_term(pred) == norm_term(gold):
                term_num_exact += 1
                term_num_ancestor += 1
            # Ancestor match (if we have a gold ID)
            elif gold_id_val and gold_id_val in term_names:
                # Get all ancestors of the gold term
                ancestors = {gold_id_val}
                if gold_id_val in ancestor_cache:
                    ancestors.update(ancestor_cache[gold_id_val])
                
                # Check if predicted term matches any ancestor term name
                pred_norm = norm_term(pred)
                for ancestor_id in ancestors:
                    if ancestor_id in term_names and norm_term(term_names[ancestor_id]) == pred_norm:
                        term_num_ancestor += 1
                        break
    
    term_acc_exact = 100.0 * term_num_exact / term_den if term_den else 0.0
    term_acc_ancestor = 100.0 * term_num_ancestor / term_den if term_den else 0.0

    # Print results
    def pct(x): return f"{x:.2f}%"
    print("\n=== ID-Based Metrics ===")
    print(f'Exact ID accuracy: {pct(id_acc_exact)} ({id_num_exact} / {id_den})')
    print(f'Ancestor-based ID accuracy: {pct(id_acc_ancestor)} ({id_num_ancestor} / {id_den})')
    
    print("\n=== Term-Based Metrics ===")
    print(f'Exact term accuracy: {pct(term_acc_exact)} ({term_num_exact} / {term_den})')
    print(f'Ancestor-based term accuracy: {pct(term_acc_ancestor)} ({term_num_ancestor} / {term_den})')
    
    print("\n=== Coverage Metrics ===")
    print(f'Coverage (Term): {pct(term_cov)} ({term_cov_n} / {total_rows})')
    print(f'Coverage (ID): {pct(id_cov)} ({id_cov_n} / {total_rows})')

if __name__ == "__main__":
    main()