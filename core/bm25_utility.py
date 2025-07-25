
from core.preprocessing_pipeline import preprocessing_pipeline
from rank_bm25 import BM25Okapi
import pandas as pd

# Load the dataset
df = pd.read_csv('data/CCSS Common Core Standards(English Standards).csv')
df.dropna(inplace=True)
df['State Standard'] = df['State Standard'].apply(lambda x: preprocessing_pipeline(x).preprocess())

# Tokenize the documents for BM25
tokenized_docs = [doc.lower().split() for doc in df['State Standard']]
bm25 = BM25Okapi(tokenized_docs)


class bm25_utility:
    def __init__(self,text,top_n=5):
        self.text = text
        self.top_n = top_n

    def retrieve_top_n_bm25(self):
        preprocessing_pipeline_instance = preprocessing_pipeline(self.text)
        preprocessed_text = preprocessing_pipeline_instance.preprocess()
        tokenized_query = preprocessed_text.split()
        
        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_n]


        # ID	Category	Sub Category	State Standard

        results = []
        for idx in top_indices:
            row = df.iloc[idx]
            results.append({
                "ID": row["ID"],
                "Category": row["Category"],
                "Sub Category": row["Sub Category"],
                "standard": row["State Standard"],
                "score": round(scores[idx], 4)

            })
        return results

query = "Identify the main idea of a text"
bm25_utility_instance = bm25_utility(query, top_n=5)
top_n_results = bm25_utility_instance.retrieve_top_n_bm25()
print(top_n_results)