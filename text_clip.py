#this script will be used to compare the embeddings of various words in the token space

from transformers import AutoTokenizer, CLIPTextModelWithProjection,CLIPModel

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import random
import nltk
from nltk.corpus import words

# Download the words corpus if not already downloaded
nltk.download('words')

# Get a list of English words
word_list = words.words()

# Get a random word
random_word = random.choice(word_list)

# Get multiple random words (e.g., 5 random words)
common_nouns = random.sample(word_list, 150)

art_nouns=["art", "painting","drawing"]

subject_nouns=["face","portrait","head"]

inputs = tokenizer(art_nouns+subject_nouns+common_nouns, padding=True, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
text_embeds=outputs.text_embeds.detach().numpy()

art_embeds=text_embeds[:len(art_nouns)]
subject_embeds=text_embeds[len(art_nouns):len(art_nouns)+len(subject_nouns)]
common_embeds=text_embeds[len(subject_nouns):]

names=["art","subject","common"]
embed_list=[art_embeds,subject_embeds, common_embeds]
for i in range(3):
    for j in range(i+1,3):
        print(names[i],names[j])
        cum=cdist(embed_list[i], embed_list[j], "euclidean")
        print('\t',np.mean(cum))

# Labels to differentiate the lists for coloring
labels = ['Medium Embeds'] * len(art_embeds) + ['Subject Embeds'] * len(subject_embeds) + ['Common Embeds'] * len(common_embeds)

# Combine all embeddings into a single array
all_embeddings = np.vstack([art_embeds, subject_embeds, common_embeds])

# Perform t-SNE to reduce embeddings to 2D
tsne = TSNE(n_components=2, random_state=42,perplexity=10)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Create a scatter plot with matplotlib
plt.figure(figsize=(8, 6))

# Assign colors for each list
colors = {
    'Medium Embeds': 'red',
    'Subject Embeds': 'green',
    'Common Embeds': 'blue'
}

# Plot each point with corresponding color
for label in np.unique(labels):
    idx = [i for i, lbl in enumerate(labels) if lbl == label]
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label, color=colors[label], s=100)

# Add plot legend and labels
plt.legend()
plt.title('t-SNE Plot of Medium, Subject, and Common Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)

# Save the plot as 'plot.png'
plt.savefig('plot.png')