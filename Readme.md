# 📚 CCSS Alignment with BM25 & SPLADE

This project allows you to align input educational text (lesson plans, learning objectives) with Common Core State Standards (ELA) using two retrieval techniques:

- **BM25** (sparse lexical search)
- **SPLADE** (sparse transformer embeddings)

## 🚀 How to Run the App

Make sure you're in the project root folder, then run:

```bash
streamlit run app.py
```

You will be able to:
- Select either BM25 or SPLADE
- Input a query (e.g., "identify key ideas and details")
- View top-matching CCSS standards
- Compare accuracy between both retrieval models

## 🧪 Sample Starter Code for app.py

```python
import streamlit as st
from core.bm25_utility import bm25_utility
from core.splade_utility import SpladeUtility

query = st.text_input("Enter your query:")
method = st.selectbox("Choose retrieval method", ["BM25", "SPLADE"])

if st.button("Get Standards"):
    if method == "BM25":
        results = bm25_utility(query).retrieve_top_n_bm25()
    else:
        results = SpladeUtility(query).retrieve_top_n_splade()

    for r in results:
        st.write(f"**{r['ID']}** - {r['standard']} (Score: {r['score']})")
```


## 📝 Notes

- Ensure that model weights for SPLADE are downloaded or cached.
- Make sure you're using cleaned and preprocessed CCSS data for accurate matching.
- Streamlit interface supports rapid switching between BM25 and SPLADE for testing.

---

**Author**: Shivendra Gupta  
**Purpose**: Educational NLP for aligning teaching content to learning standards.