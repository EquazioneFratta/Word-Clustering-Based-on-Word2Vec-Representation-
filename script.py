import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def visualize_vectors(model, words, save_path):
    vectors = [model.wv[w] for w in words]
    labels = [w for w in words]
    
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)
    
    plt.figure(figsize=(8,6))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])  
    
    for i, word in enumerate(labels):
        plt.annotate(word,
                     (reduced_vectors[i, 0], reduced_vectors[i, 1]),
                     xytext=(5, 2),
                     textcoords='offset points'
                     )

    plt.title("Vettori Word2Vec con PCA")
    plt.savefig(save_path)
    
def explore_model(model, words_to_explore):
    for word in words_to_explore:
        
        print(f"Parole più simili a '{word}':")
        similar_words = model.wv.most_similar(word, topn=5)
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.4f}")
        
    return [w for w in words_to_explore if w in model.wv]

def cluster_words(vectors, n_clusters=3):
    
    if len(vectors) < n_clusters:
        if len(vectors) < 2:
            print("Non ci sono abbastanza parole valide per eseguire il clustering.")
            return None, None, None
        else:
            print(f"Riduzione del numero di cluster da {n_clusters} a {len(vectors)}")
            n_clusters = len(vectors)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    
    return labels

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
    
    model = Word2Vec.load("w2v_model.model")
    
    print("Esplorazione del modello ")
    default_words = ["man", "woman", "love", "death", "life", "time"]
    
    print("Inserisci le parole da esplorare (separate da virgola) o premi Invio per i valori suggeriti:")
    print(f"Parole suggerite: {', '.join(default_words)}")
    words_input = input("> ").strip()
    
    if words_input:
        words_to_explore = [w.strip() for w in words_input.split(",")]
    else:
        words_to_explore = default_words
        print(f"Uso parole predefinite: {', '.join(default_words)}")
    valid_words = []
    missing_words = []
    vectors = []
    for word in words_to_explore:
        word_lower = word.lower()
        if word_lower in model.wv:
            vectors.append(model.wv[word_lower])
            valid_words.append(word_lower)
        else:
            missing_words.append(word)
    
    if missing_words:
        print(f"Parole non trovate nel vocabolario: {missing_words}")

    valid_words = explore_model(model, valid_words)
    
    print("Visualizzazione vettori")
    visualize_vectors(model, valid_words, save_path="word_vectors_pca.png")
            
    print("Clustering delle parole")
    
    
    n_clusters = int(input("Inserisci il numero di cluster (predefinito: 3): ") or 3)
    
    labels = cluster_words(vectors, n_clusters)
    
    visualize_clusters(
        valid_words, vectors, labels,
        title=f"Clustering delle parole con Word2Vec + KMeans - {n_clusters} cluster",
        save_path="word_clusters_pca.png"
    )