import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F

model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
df = pd.read_csv('/Users/shivendragupta/Desktop/internship25/CCSS/data/CCSS Common Core Standards(English Standards).csv')
df.dropna(inplace=True)

# Reset index to align doc IDs

class splade_utility:
    def __init__(self, query, top_n=5):
        self.query = query
        self.top_n = top_n
        
    @staticmethod
    def get_splade_sparse_vector(text):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            logits = model(**inputs).logits.squeeze(0)  # [seq_len, vocab_size]
            relu_out = F.relu(logits)
            splade_weights = torch.log1p(relu_out).max(dim=0).values
            indices = torch.nonzero(splade_weights).squeeze()
            return {
                tokenizer.convert_ids_to_tokens([i.item()])[0]: splade_weights[i].item()
                for i in indices
            }
        
    def dot_product_sparse(self , query_vec, doc_vec):
        return sum(query_vec.get(term, 0.0) * doc_vec.get(term, 0.0) for term in query_vec)

    def retrieve_top_n_splade(self):
        query_vec = self.get_splade_sparse_vector(self.query)
        scores = [
            (self.dot_product_sparse(query_vec, doc_vec), idx)
            for idx, doc_vec in enumerate(splade_doc_vectors)
        ]
        
        top_matches = sorted(scores, reverse=True)[:self.top_n]
        
        results = []
        for score, idx in top_matches:
            results.append({
                "score": round(score, 4),
                "standard": df.iloc[idx]["State Standard"],
                "ID": df.iloc[idx]["ID"],
                "Category": df.iloc[idx]["Category"],
                "Sub Category": df.iloc[idx]["Sub Category"]
            })
        return results

df = df.reset_index(drop=True)

# Get list of standard texts
standard_texts = df["State Standard"].astype(str).tolist()

# Compute sparse vectors
splade_doc_vectors = [splade_utility.get_splade_sparse_vector(text) for text in (standard_texts)]


# Example usage
query = "determine main idea text explain supported key detail summarize text"
splade_instance = splade_utility(query)
results = splade_instance.retrieve_top_n_splade()
print(results)