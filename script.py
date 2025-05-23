import re
import string
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

def visualize_vectors(model, words, save_path=None):
    vectors = []
    labels = []
    missing_words = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in model.wv:
            vectors.append(model.wv[word_lower])
            labels.append(word_lower)
        else:
            missing_words.append(word)
    
    if missing_words:
        print(f"Parole non trovate nel vocabolario: {missing_words}")
    
    if len(vectors) < 2:
        print("Non ci sono abbastanza parole nel vocabolario per visualizzare.")
        return
    
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], s=100)
    
    for i, word in enumerate(labels):
        plt.annotate(word, 
                    (reduced_vectors[i, 0], reduced_vectors[i, 1]),
                    fontsize=12,
                    alpha=0.8,
                    xytext=(5, 2),
                    textcoords='offset points')
    
    plt.title("Visualizzazione vettori di parole (PCA)")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def download_gutenberg_book(url_or_id):
    #TODO: estrare l'ID del libro da un URL e poi sotto ricrea l'URL con l'ide estratto, mi sembra una cosa idiota
    # se l'input è un URL, estrai l'ID del libro
    if isinstance(url_or_id, str) and url_or_id.startswith('http'):
        match = re.search(r'/(\d+)/?$', url_or_id)
        if match:
            book_id = match.group(1)
        else:
            raise ValueError("Impossibile estrarre l'ID del libro dall'URL")
    else:
        book_id = str(url_or_id)
    
    # Construct the download URL for the plain text version
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError: #TODO: sta cosa dell'url alternativo non mi convince
        # url alternativo
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        response = requests.get(url)
        response.raise_for_status()
        return response.text

def preprocess_text(raw_text):
    raw_text = raw_text.lower()
    
    raw_text = re.sub(r'\d+', '', raw_text)
    raw_text = raw_text.translate(str.maketrans('', '', string.punctuation))
    
    stop_words = set(stopwords.words("english"))
    
    sentences = []
    raw_sentences = re.split(r'(?<=[.!?])\s+', raw_text)
    
    print(f"Frasi totali trovate: {len(raw_sentences)}")
    print("Elaborazione delle frasi in corso...")
    
    for i, sentence in enumerate(raw_sentences):
        words = [word for word in word_tokenize(sentence) if word.isalpha() and word not in stop_words]
        if words:  # aggiungi solo frasi non vuote
            sentences.append(words)

    return sentences

def train_word2vec(sentences, vector_size=100, window=5, min_count=5, sg=1):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model

def explore_model(model, words_to_explore=None):
    if words_to_explore is None:
        common_words = ["man", "woman", "sea", "ship", "love", "time"]
        words_to_explore = [word for word in common_words if word.lower() in model.wv]
        
        if not words_to_explore and len(model.wv.index_to_key) > 0:
            words_to_explore = model.wv.index_to_key[:5]
    
    print("\n=== Esplorazione Modello Word2Vec ===")
    for word in words_to_explore:
        word_lower = word.lower()
        if word_lower in model.wv:
            print(f"\nParole più simili a '{word_lower}':")
            similar_words = model.wv.most_similar(word_lower, topn=5)
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        else:
            print(f"\nParola '{word}' non presente nel vocabolario.")
    
    return [w.lower() for w in words_to_explore if w.lower() in model.wv]  # Return valid words for visualization

def cluster_words(model, words_to_cluster, n_clusters=3):
    """
    Cluster a list of words using KMeans based on their Word2Vec representations.
    """
    # Check if words are in the vocabulary (case insensitive)
    vectors = []
    valid_words = []
    missing_words = []
    
    for word in words_to_cluster:
        word_lower = word.lower()
        if word_lower in model.wv:
            vectors.append(model.wv[word_lower])
            valid_words.append(word_lower)
        else:
            missing_words.append(word)
    
    if missing_words:
        print(f"Parole non trovate nel vocabolario: {missing_words}")
    
    if len(vectors) < n_clusters:
        if len(vectors) < 2:
            print("Non ci sono abbastanza parole valide per eseguire il clustering.")
            return None, None, None
        else:
            print(f"Riduzione del numero di cluster da {n_clusters} a {len(vectors)}")
            n_clusters = len(vectors)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    
    return valid_words, vectors, labels

def visualize_clusters(valid_words, vectors, labels, title, save_path=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    
    plt.figure(figsize=(10, 7))
    n_clusters = len(set(labels))
    palette = sns.color_palette("bright", n_clusters)
    
    for i, word in enumerate(valid_words):
        plt.scatter(reduced[i, 0], reduced[i, 1], color=palette[labels[i]], s=100)
        plt.text(reduced[i, 0] + 0.002, reduced[i, 1] + 0.002, word, fontsize=12)
    
    plt.title(title)
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

if __name__ == "__main__":
    print("Addestramento Word2Vec e Clustering")
    print("==================================")
    print("\nScegli una fonte di testo:")
    print("1. Carica file locale 'moby-dick-clean.txt'")
    print("2. Scarica un libro da Project Gutenberg")

    choice = input("\nInserisci la tua scelta (1-2): ").strip()
    
    if choice == "1":
        # Load from local file
        try:
            with open("moby-dick-clean.txt", "r", encoding="utf-8") as f:
                raw_text = f.read()
            print("'moby-dick-clean.txt' caricato con successo")
        except FileNotFoundError:
            print("File 'moby-dick-clean.txt' non trovato. Assicurati che esista nella directory corrente.")
            exit()
    
    elif choice == "2":
        # Download from Project Gutenberg
        book_id = input("Inserisci l'ID del libro su Project Gutenberg (es. 2701 per Moby Dick): ").strip()
        print(f"Download del libro da Project Gutenberg...")
        try:
            raw_text = download_gutenberg_book(book_id)
            print("Download completato!")
        except Exception as e:
            print(f"Errore durante il download del libro: {str(e)}")
            exit()
    else:
        print("Scelta non valida. Uscita...")
        exit()
    
    # Preprocess the text
    print("\nPreelaborazione del testo...")
    sentences = preprocess_text(raw_text)
    print(f"Testo preelaborato: {len(sentences)} frasi estratte")
    
    # Train Word2Vec model
    print("\nAddestramento modello Word2Vec...")
    model = train_word2vec(sentences)
    print("Addestramento del modello completato!")
    
    # Save the model
    model_path = "w2v_model.model"
    model.save(model_path)
    print(f"Modello salvato in {model_path}")
    
    # Explore the model
    print("\n--- Esplorazione del modello ---")
    default_words = ["man", "woman", "love", "death", "life", "time"]
    
    print("\nInserisci le parole da esplorare (separate da virgola) o premi Invio per i valori suggeriti:")
    print(f"Parole suggerite: {', '.join(default_words)}")
    words_input = input("> ").strip()
    
    if words_input:
        words_to_explore = [w.strip() for w in words_input.split(",")]
    else:
        words_to_explore = default_words
        print(f"Uso parole predefinite: {', '.join(default_words)}")
    
    # Explore and get valid words
    valid_words = explore_model(model, words_to_explore)
    
    if valid_words and len(valid_words) >= 2:
        print("\n--- Visualizzazione vettori ---")
        visualize_vectors(model, valid_words, save_path="word_vectors_pca.png")
        
        input("\nPremi Invio per continuare al clustering...")
    else:
        print("\nNon ci sono abbastanza parole valide per visualizzare i vettori.")
        exit()
    
    # Cluster words
    print("\n--- Clustering delle parole ---")
    default_clusters = ["man", "woman", "love", "death", "life", "time", "world", "water"]
    
    print("\nInserisci le parole da raggruppare (separate da virgola) o premi Invio per usare le stesse parole:")
    print(f"Parole suggerite: {', '.join(default_clusters)}")
    cluster_input = input("> ").strip()
    
    if cluster_input:
        words_to_cluster = [w.strip() for w in cluster_input.split(",")]
    else:
        # Use the same words from exploration if available, otherwise use defaults
        words_to_cluster = valid_words if valid_words else default_clusters
        print(f"Uso parole: {', '.join(words_to_cluster)}")
    
    n_clusters = int(input("\nInserisci il numero di cluster (predefinito: 3): ") or 3)
    
    valid_words, vectors, labels = cluster_words(model, words_to_cluster, n_clusters)
    
    if valid_words and vectors is not None and labels is not None:
        visualize_clusters(
            valid_words, vectors, labels,
            title=f"Clustering delle parole con Word2Vec + KMeans - {n_clusters} cluster",
            save_path="word_clusters_pca.png"
        )
        print(f"\nVisualizzazione salvata in word_clusters_pca.png")
    else:
        print("\nImpossibile creare clusters con le parole specificate.")
        print("Suggerimento: prova con parole più comuni nel testo o in minuscolo.")