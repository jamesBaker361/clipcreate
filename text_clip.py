#this script will be used to compare the embeddings of various words in the token space

from transformers import AutoTokenizer, CLIPTextModelWithProjection,CLIPModel

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

common_nouns = [
    "dog", 
    "cat", 
    "car", 
    "house", 
    "book", 
    "tree", 
    "chair", 
    "table", 
    "phone", 
    "water", 
    "city", 
    "food", 
    "school", 
    "friend", 
    "flower",
      "computer",
    "mountain",
    "river",
    "window",
    "desk"
]

art_nouns=["art", "painting","drawing"]

subject_nouns=["man","woman","person"]

inputs = tokenizer(art_nouns+subject_nouns+common_nouns, padding=True, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
text_embeds=outputs.text_embeds.detach().numpy()

art_embeds=text_embeds[:len(art_nouns)]
subject_embeds=text_embeds[len(art_nouns):len(art_nouns)+len(subject_nouns)]
common_embeds=text_embeds[len(subject_nouns):]