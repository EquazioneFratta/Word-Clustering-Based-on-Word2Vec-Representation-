import unicodedata
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')

# preprocess the text through tokenization, , and stopwords .
def preprocess_text(raw_text):
    # lowercasing
    raw_text = raw_text.lower()
    # punctuation removal
    # normalizza in NFKD (separa lettere e segni diacritici)
    raw_text = unicodedata.normalize("NFKD", text)
    # rimuove simboli non alfabetici (compresi ’ “ … ecc.)
    raw_text = re.sub(r"[^\w\s]", " ", text)
    
    stop_words = set(stopwords.words("english"))
    
    sentences = []
    raw_sentences = re.split(r'(?<=[.!?])\s+', raw_text)
    
    # print(f"Frasi totali trovate: {len(raw_sentences)}")
    
    for sentence in raw_sentences:
        # word.isalpha() → tiene solo parole fatte di lettere (quindi esclude numeri, simboli, ecc.). Secondo me stringe troppo
        # words = [word for word in word_tokenize(sentence) if word.isalpha() and word not in stop_words] 
        words = [word for word in word_tokenize(sentence) if word not in stop_words]
    
        if words:
            sentences.append(words)

    return sentences

if __name__ == "__main__":
    all_texts = []
    folder_path = "corpus"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                all_texts.append(raw_text)
    
    # Preprocessa ogni file e accumula le frasi
    all_sentences = []
    for text in all_texts:
        sentences = preprocess_text(text)
        all_sentences.extend(sentences)
        
    print("\nAddestramento modello Word2Vec...")
    # slide 99-100 spiega i parametri
    # min_count : Tutte le parole con frequenza < 5 vengono ignorate e non entrano nel vocabolario del modello
    model = Word2Vec(all_sentences, vector_size=300, window=15, min_count=2, sg=1, epochs=50)
    
    print("Addestramento del modello completato!")
    
    model.save("w2v_model.model")
    