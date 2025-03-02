import transformers
import torch
import pandas as pd
import random
from tqdm import tqdm

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
pipeline_generator = transformers.pipeline(
    task="text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

topics = [
    "technology trends", "sports events", "daily routines", "weather updates",
    "book recommendations", "neutral news commentary", "travel experiences",
    "health tips", "work-life balance", "general facts"
]

def make_prompt(topic):
    return (
        " {topic}"


    )

def clean_text(text):
    # Jeśli chcesz usuwać jakieś frazy:
    to_remove = [
        "Write a neutral social media post about",
        "Keep it brief and informative.",
        "Do not include any headings such as '' or ''."
    ]
    for fragment in to_remove:
        text = text.replace(fragment, "")
    return text.strip()

def generate_posts_batch(n):
    """
    Generuje n postów w jednej paczce (batch),
    zwracając listę tekstów wygenerowanych przez model.
    """
    prompts = [make_prompt(random.choice(topics)) for _ in range(n)]
    # Pipeline w wersji batch - wystarczy przekazać listę promptów
    outputs = pipeline_generator(prompts, max_length=50, do_sample=True, temperature=0.7)
    # outputs będzie listą o tej samej długości co prompts
    # Każdy element to słownik z kluczem 'generated_text'
    generated_texts = [clean_text(o[0]['generated_text']) for o in outputs]
    return generated_texts

num_posts = 10
batch_size = 8  # dobierz do pojemności pamięci i wydajności
posts = []

num_batches = num_posts // batch_size
remainder = num_posts % batch_size

# Generowanie partiami
for _ in tqdm(range(num_batches), desc="Generating posts in batches"):
    batch_posts = generate_posts_batch(batch_size)
    posts.extend(batch_posts)

# Jeśli coś zostało "nadmiarowego"
if remainder > 0:
    last_posts = generate_posts_batch(remainder)
    posts.extend(last_posts)

# Zapis do CSV
df = pd.DataFrame(posts, columns=["post"])
df.to_csv("neutral_posts.csv", index=False, encoding="utf-8")

print("✅ Neutral posts saved to 'neutral_posts.csv'")
