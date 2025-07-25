import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

from core.splade_utility import splade_utility
from core.bm25_utility import bm25_utility



# ==== Import your models and functions ====
# Assume these are already defined in your notebook/script
# from your_code import retrieve_top_n_bm25, retrieve_top_n_splade, evaluate_top1_on_state_standard

# Dummy accuracy values (replace with real ones from your code)
bm25_accuracy = 0.9959
splade_accuracy = 0.9797

# Dummy placeholder functions (replace with your actual ones)
def retrieve_top_n_bm25(query, top_n=5):
    bm25_utility_instance = bm25_utility(query, top_n=5)
    top_n_results = bm25_utility_instance.retrieve_top_n_bm25()
    return top_n_results

def retrieve_top_n_splade(query, top_n=5):
    splade_utility_instance = splade_utility(query, top_n=top_n)
    return splade_utility_instance.retrieve_top_n_splade()

# ==== Streamlit UI ====

st.set_page_config(page_title="CCSS Alignment", layout="centered")
st.title("📚 CCSS Alignment Search")

# Select model
model_choice = st.radio("Select Retrieval Model:", ["BM25", "SPLADE"])

# Accuracy bar chart
st.subheader("🎯 Model Top-1 Accuracy")
fig, ax = plt.subplots()
ax.bar(["BM25", "SPLADE"], [bm25_accuracy, splade_accuracy], color=["skyblue", "lightgreen"])
ax.set_ylim([0.9, 1.01])
ax.set_ylabel("Top-1 Accuracy")
for i, acc in enumerate([bm25_accuracy, splade_accuracy]):
    ax.text(i, acc + 0.001, f"{acc:.4f}", ha='center', fontsize=10)
st.pyplot(fig)

# Query input
st.subheader("🔍 Try a Query")
query = st.text_area("Enter a lesson or objective text:", height=100)

# Search button
if st.button("Search"):
    st.subheader("📄 Top Results")

    if model_choice == "BM25":
        results = retrieve_top_n_bm25(query, top_n=5)
    else:
        results = retrieve_top_n_splade(query, top_n=5)

    if results:
        for i, r in enumerate(results, 1):
            st.markdown(f"""
            **Rank {i}**  
            - **Standard**: {r['standard']}  
            - **ID**: {r.get('ID', 'N/A')}  
            - **Category**: {r.get('Category', 'N/A')}  
            - **Sub Category**: {r.get('Sub Category', 'N/A')}  
            - **Score**: `{r['score']}`
            """)
    else:
        st.warning("No results found.")
