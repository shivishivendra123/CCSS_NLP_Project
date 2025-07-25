import re
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))

class preprocessing_pipeline:
    def __init__(self,text):
        self.text = text

    def preprocess(self):
        self.text = self.clean_text(self.text)
        self.text = self.lowercase(self.text)
        self.text = self.remove_punctuation(self.text)
        self.text = self.remove_stopwords(self.text)
        self.text = self.lemmatize_tokens(self.text)
        return self.text
    
    def clean_text(self , text: str) -> str:
        text = text.strip()
        text = text.replace("\n", " ").replace("\xa0", " ")
        text = text.replace("“", "\"").replace("”", "\"").replace("–", "-")
        return text
    
    def lowercase(self, text: str) -> str:
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)
    
    def remove_stopwords(self, text: str) -> str:
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word not in stop_words])
    
    def lemmatize_tokens(self, text: str) -> str:
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    

