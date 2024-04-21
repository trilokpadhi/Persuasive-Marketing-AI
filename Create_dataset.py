
#!pip install hnswlib sentence-transformers

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# from google.colab import drive
# drive.mount('/content/drive')
# data_dir = "/content/drive/MyDrive/Agnik/ADM_Assignment"

import pandas as pd
data = pd.read_csv(f"/content/drive/MyDrive/ADM_Assignment/data.csv")

print(len(data))



model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = data['headline_text'].tolist()
# Create embeddings

mini_sentences = sentences[:90000]

embeddings = model.encode(mini_sentences)

def calculate_average_length(texts):
    total_length = 0
    num_texts = len(texts)

    for text in texts:
        total_length += len(str(text))

    if num_texts > 0:
        average_length = total_length / num_texts
        return average_length
    else:
        return 0

# Example usage
avg_length = calculate_average_length(sentences)
print(f"Average length of texts: {avg_length}")

from statistics import mode

def calculate_mode_length(texts):
    text_lengths = [len(str(text)) for text in texts]
    mode_length = mode(text_lengths)
    return mode_length

# Example usage
mode_length = calculate_mode_length(sentences)
print(f"Mode Length: {mode_length}")

import matplotlib.pyplot as plt

def plot_text_length_histogram(texts):
    text_lengths = [len(str(text)) for text in texts]

    plt.figure(figsize=(8, 6))
    plt.hist(text_lengths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Text Lengths')
    plt.grid(True)
    plt.show()

# Example usage
plot_text_length_histogram(sentences)

# Dimension of our vector space
dimension = embeddings.shape[1]

# Create a new index
p = hnswlib.Index(space='cosine', dim=dimension)

# Initialize an index - the maximum number of elements should be known beforehand
p.init_index(max_elements=100000, ef_construction=200, M=16)

# Element insertion (can be called several times)
p.add_items(embeddings)

# Controlling the recall by setting ef:
p.set_ef(50) # ef should always be > k

new_sentence = "A persuasive advertisement for nike shoes"
new_embedding = model.encode([new_sentence])

# Fetch k neighbors
labels, distances = p.knn_query(new_embedding, k=5)

print(labels)

new_sentence = "We can't wait to see you ?"
new_embedding = model.encode([new_sentence])

# Fetch k neighbors
labels, distances = p.knn_query(new_embedding, k=30)

print(labels)

sentences_m = []

for i in range(len(labels[0])):
  print(sentences[labels[0][i]])
  sentences_m.append(sentences[labels[0][i]])


start_time = time.time()

# Repeat the search process to get a fair measure
for _ in range(1000):
    labels, distances = p.knn_query(new_embedding, k=5)

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time for 1000 searches: ", elapsed_time)

