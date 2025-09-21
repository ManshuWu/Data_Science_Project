import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz

# ------------------------------
# Define paths for input data
# ------------------------------
# Path to the pre-trained NER model directory (not used in this script but kept for reference)
MODEL_DIR = r"C:\Users\13189\Desktop\ner_pubmedbert_saved_HPO"

# Path to the test dataset (JSONL file)
DEF_TEST = r"C:\Users\13189\Desktop\bio_outputs\test.jsonl"

# URL and path for the HPO OBO file
DEF_OBO_URL = "http://purl.obolibrary.org/obo/hp.obo"
DEF_OBO_PATH = "hp.obo"

# ------------------------------
# 0) Parse the OBO file and build the dictionary
# ------------------------------
def parse_obo(path: str) -> Dict[str, Dict]:
    """Parse OBO file and extract term information"""
    terms = {}
    current_term = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                if current_term and current_term.get("id"):
                    terms[current_term["id"]] = current_term
                current_term = {"id": None, "name": None, "is_a": [], "synonym": [], "def": None}
            elif line.startswith("id: "):
                current_term["id"] = line[4:]
            elif line.startswith("name: "):
                current_term["name"] = line[6:]
            elif line.startswith("is_a: "):
                current_term["is_a"].append(line[6:].split("!")[0].strip())
            elif line.startswith("synonym: "):
                current_term["synonym"].append(line[9:])
            elif line.startswith("def: "):
                current_term["def"] = line[5:]
    
    if current_term and current_term.get("id"):
        terms[current_term["id"]] = current_term
    
    return terms

def obo_name(terms: Dict, term_id: str) -> str:
    """Get the name of a term by its ID"""
    return terms.get(term_id, {}).get("name", "") or ""

def obo_syns(terms: Dict, term_id: str) -> List[str]:
    """Get all synonyms of a term by its ID"""
    synonyms = []
    for synonym in terms.get(term_id, {}).get("synonym", []):
        match = re.search(r'"(.*?)"', synonym)
        if match:
            synonyms.append(match.group(1))
        else:
            synonyms.append(synonym)
    return list(dict.fromkeys([s for s in synonyms if s]))

# ------------------------------
# Text Encoding Functions using Sentence-Transformers
# ------------------------------
def get_sentence_model():
    """Initialize and return the sentence transformer model"""
    model_name = "sentence-transformers/pubmedbert-base-uncased"
    model = SentenceTransformer(model_name)
    return model

def encode_text(text: str, model: SentenceTransformer) -> np.ndarray:
    """Normalize and encode text into embeddings"""
    text = text.lower().strip()
    embeddings = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

# ------------------------------
# Dictionary Matching and Fuzzy Search Strategy
# ------------------------------
class HPORecommender:
    def __init__(self, obo_file: str):
        """Initialize HPO recommender with OBO file"""
        # Download OBO file if it doesn't exist
        if not os.path.exists(obo_file):
            print(f"Downloading HPO OBO file from {DEF_OBO_URL}...")
            import urllib.request
            urllib.request.urlretrieve(DEF_OBO_URL, obo_file)
        
        # Parse OBO file
        self.terms = parse_obo(obo_file)
        self.lexicon = defaultdict(set)
        self.surface_index = []
        
        # Build lexicon and surface index
        for term_id in self.terms:
            if not term_id.startswith("HP:"):
                continue
            texts = [obo_name(self.terms, term_id)] + obo_syns(self.terms, term_id)
            for text in set([t for t in texts if t]):
                normalized_text = self.normalize_surface(text)
                self.lexicon[normalized_text].add(term_id)
                self.surface_index.append((text, term_id))
        
        print(f"Built lexicon with {len(self.lexicon)} unique surface forms")
        print(f"Surface index contains {len(self.surface_index)} entries")
        
        # Load the Sentence-Transformer model for text encoding
        self.model = get_sentence_model()
        
        # Pre-encode all terms for efficient vector search
        print("Pre-encoding all HPO terms...")
        self.encoded_terms = []
        for text, term_id in self.surface_index:
            embedding = encode_text(text, self.model)
            self.encoded_terms.append((term_id, text, embedding))
        print("Pre-encoding completed")

    def normalize_surface(self, text: str) -> str:
        """Normalize text for matching"""
        text = re.sub(r"\s+", " ", (text or "").lower()).strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def exact_lookup(self, mention: str) -> List[str]:
        """Find exact matches for a mention"""
        normalized_mention = self.normalize_surface(mention)
        return list(self.lexicon.get(normalized_mention, set()))

    def fuzzy_candidates(self, mention: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Find fuzzy matches for a mention"""
        choices = [text for text, _ in self.surface_index]
        matches = process.extract(mention, choices, scorer=fuzz.WRatio, limit=top_k*3)
        
        best_matches = {}
        for matched_text, score, index in matches:
            hpo_id = self.surface_index[index][1]
            if hpo_id not in best_matches or score > best_matches[hpo_id][1]:
                best_matches[hpo_id] = (matched_text, score)
        
        results = [(hpo_id, text, score) for hpo_id, (text, score) in best_matches.items()]
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def vector_search(self, mention: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find semantic matches using vector similarity"""
        mention_embedding = encode_text(mention, self.model)
        similarities = []
        
        for term_id, text, term_embedding in self.encoded_terms:
            cosine_sim = np.dot(mention_embedding, term_embedding)
            similarities.append((term_id, cosine_sim, text))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(term_id, score) for term_id, score, _ in similarities[:top_k]]

    def get_hpo_id(self, mention: str) -> Tuple[str, str, float, str]:
        """Get the best HPO ID for a mention using a cascading approach"""
        # Stage 1: Exact dictionary matching
        exact_matches = self.exact_lookup(mention)
        if exact_matches:
            hpo_id = exact_matches[0]
            hpo_name = obo_name(self.terms, hpo_id)
            return hpo_id, hpo_name, 1.0, "exact"
        
        # Stage 2: Fuzzy matching
        fuzzy_matches = self.fuzzy_candidates(mention, top_k=1)
        if fuzzy_matches and fuzzy_matches[0][2] > 80:  # Threshold for fuzzy matching
            hpo_id, matched_text, score = fuzzy_matches[0]
            hpo_name = obo_name(self.terms, hpo_id)
            confidence = score / 100.0  # Convert from 0-100 to 0.0-1.0
            return hpo_id, hpo_name, confidence, "fuzzy"
        
        # Stage 3: Vector-based semantic search
        vector_matches = self.vector_search(mention, top_k=1)
        if vector_matches and vector_matches[0][1] > 0.7:  # Threshold for vector matching
            hpo_id, similarity = vector_matches[0]
            hpo_name = obo_name(self.terms, hpo_id)
            return hpo_id, hpo_name, similarity, "vector"
        
        return None, None, 0.0, "none"

# ------------------------------
# Process test set and generate results
# ------------------------------
def extract_hpo_mentions(tokens: List[str], labels: List[str]) -> List[str]:
    """Extract HPO mentions from BIO-formatted tokens and labels"""
    mentions = []
    current_mention = []
    
    for i, label in enumerate(labels):
        if label.startswith("B-HPO_TERM"):
            # Start a new mention
            if current_mention:
                mentions.append(" ".join(current_mention).replace(" ##", ""))
                current_mention = []
            current_mention.append(tokens[i])
        elif label.startswith("I-HPO_TERM") and current_mention:
            # Continue current mention
            current_mention.append(tokens[i])
        elif current_mention:
            # End of mention
            mentions.append(" ".join(current_mention).replace(" ##", ""))
            current_mention = []
    
    # Add the last mention if exists
    if current_mention:
        mentions.append(" ".join(current_mention).replace(" ##", ""))
    
    return mentions

def process_test_set(test_file: str, recommender: HPORecommender) -> pd.DataFrame:
    """Process the test set and generate normalization results"""
    results = []
    
    with open(test_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                doc_id = data.get("id", f"line_{line_num}")
                tokens = data.get("tokens", [])
                labels = data.get("labels", [])
                
                # Extract HPO mentions from BIO format
                hpo_mentions = extract_hpo_mentions(tokens, labels)
                
                for mention in hpo_mentions:
                    hpo_id, hpo_name, confidence, method = recommender.get_hpo_id(mention)
                    results.append({
                        "doc_id": doc_id,
                        "mention": mention,
                        "hpo_id": hpo_id,
                        "hpo_name": hpo_name,
                        "confidence": confidence,
                        "method": method
                    })
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line {line_num}: {e}")
    
    return pd.DataFrame(results)

# ------------------------------
# Main execution
# ------------------------------
def main():
    """Main function to run the HPO normalization pipeline"""
    # Initialize the HPO recommender
    print("Initializing HPO recommender...")
    recommender = HPORecommender(DEF_OBO_PATH)
    
    # Process the test set
    print("Processing test set...")
    results_df = process_test_set(DEF_TEST, recommender)
    
    # Save results to CSV
    output_file = "hpo_normalization_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    print("\n=== Normalization Results Summary ===")
    print(f"Total mentions processed: {len(results_df)}")
    
    successful_matches = results_df[results_df["hpo_id"].notna()]
    print(f"Successfully normalized: {len(successful_matches)}")
    
    method_counts = successful_matches["method"].value_counts()
    for method, count in method_counts.items():
        print(f"  - {method}: {count}")
    
    print(f"Failed to normalize: {len(results_df) - len(successful_matches)}")
    
    # Display first few results
    print("\n=== First 10 Results ===")
    print(successful_matches.head(10).to_string(index=False))

if __name__ == "__main__":
    main()