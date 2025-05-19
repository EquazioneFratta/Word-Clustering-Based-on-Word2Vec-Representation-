import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

### 1. Loading and Preprocessing Text
try:
    with open("moby-dick-clean.txt", "r", encoding="utf-8") as f:
        raw_text = f.read().lower()
except FileNotFoundError:
    print("File 'moby-dick-clean.txt' not found. Please make sure it exists in the current directory.")
    exit(1)

# Remove punctuation and numbers
raw_text = re.sub(r'\d+', '', raw_text)
raw_text = raw_text.translate(str.maketrans('', '', string.punctuation))

# Tokenization into words (not using sent_tokenize which causes the error)
stop_words = set(stopwords.words("english"))
# Split by whitespace instead of using sent_tokenize
sentences = []
# Simple sentence splitting by common sentence terminators
raw_sentences = re.split(r'(?<=[.!?])\s+', raw_text)
for sentence in raw_sentences:
    words = [word for word in word_tokenize(sentence) if word.isalpha() and word not in stop_words]
    if words:  # Only add non-empty sentences
        sentences.append(words)

### 2. Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1)
model.save("w2v_model.model")

### 3. Choose words to cluster
words = ["whale", "sea", "ship", "captain", "harpoon", "fish", "ocean", "water", "storm", "crew", "death", "man", "god", "revenge"]

# Check if all words are in the vocabulary
vectors = []
valid_words = []
missing_words = []
for word in words:
    if word in model.wv:
        vectors.append(model.wv[word])
        valid_words.append(word)
    else:
        missing_words.append(word)
if missing_words:
    print(f"Words not found in Word2Vec vocabulary: {missing_words}")

if len(vectors) < 2:
    print("Not enough valid words to perform clustering. Please check your input file and word list.")
    exit(1)

### 4. Clustering with KMeans
n_clusters = min(3, len(vectors))  # Ensure we don't request more clusters than data points
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(vectors)

### 5. PCA for dimension reduction to 2D
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

### 6. Visualization
plt.figure(figsize=(10, 7))
palette = sns.color_palette("bright", n_clusters)

for i, word in enumerate(valid_words):
    plt.scatter(reduced[i, 0], reduced[i, 1], color=palette[labels[i]], s=100)
    plt.text(reduced[i, 0] + 0.002, reduced[i, 1] + 0.002, word, fontsize=12)

plt.title("Word Clustering with Word2Vec + KMeans (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig("word_clusters.png")  # Save the figure as a file
plt.show()  # Display the figure (if running in an environment with GUI)