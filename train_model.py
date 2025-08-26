import unicodedata
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(raw_text):
    text = raw_text.lower()
    #per normalizzare lettere accentate
    text = unicodedata.normalize("NFKD", text)
    
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    
    processed_sentences = []
    for sentence in raw_sentences:
        #rimuove simboli speciali
        sentence = re.sub(r"[^\w\s']", " ", sentence)
        words = word_tokenize(sentence)
        #Rimuove underscore
        words = [word.strip('_') for word in words] 

        stop_words = set(stopwords.words("english"))
        #rimuove stopword e parole più corte di 2
        words = [word for word in words if word not in stop_words and len(word) > 2]
        #se la frase ha men o di 3 parole la esclude, altrimenti non ho contesto
        if len(words) > 3:  
            processed_sentences.append(words)
    
    return processed_sentences

if __name__ == "__main__":
    all_texts = []
    folder_path = "corpus"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                all_texts.append(raw_text)

    all_sentences = []
    for text in all_texts:
        sentences = preprocess_text(text)
        all_sentences.extend(sentences)

    print("Addestramento modello ...")
    #vector size: determina quante dimensioni ha ogni vettore di parola
    #window: quante parole a sinistra e destra considera per ogni parola target
    #min_count: ignora parole che appaiono meno di N volte nel corpus
    model = Word2Vec(all_sentences, vector_size=200, window=7, min_count=4, sg=1, epochs=50)
    
    print("Addestramento del modello completato")
    
    model.save("w2v_model.model")
    