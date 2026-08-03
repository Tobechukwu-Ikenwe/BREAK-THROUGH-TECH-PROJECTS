# === Imports ===
from openai import OpenAI
from scipy.spatial.distance import cosine
from IPython.display import display, Markdown

# === OpenAI Client Setup ===
client = OpenAI()

# === Example Texts ===
text_0 = "I like football"  # Duplicate of text_1 (to demonstrate perfect similarity)
text_1 = "I like football"
text_2 = "Sports"
text_3 = "Arts"

# === Create embeddings for all texts in a single API call ===
response = client.embeddings.create(
    model="text-embedding-3-small", 
    input=[text_0, text_1, text_2, text_3]
)

# === Extract the embedding vectors (lists of floats) ===
v0 = response.data[0].embedding  # embedding for text_0
v1 = response.data[1].embedding  # embedding for text_1
v2 = response.data[2].embedding  # embedding for text_2
v3 = response.data[3].embedding  # embedding for text_3

# === Display results ===
output = f"""
**Embedding Dimensionality:**
- '{text_0}': {len(v0)} dimensions
- '{text_1}': {len(v1)} dimensions
- '{text_2}': {len(v2)} dimensions
- '{text_3}': {len(v3)} dimensions

---

### Vector Preview (Text 0)
**Text:** {text_0}  
**First 5 values:** `{v0[:5]}`  
**Last 5 values:** `{v0[-5:]}`

---

### Vector Preview (Text 1)
**Text:** {text_1}  
**First 5 values:** `{v1[:5]}`  
**Last 5 values:** `{v1[-5:]}`

---

### Vector Preview (Text 2)
**Text:** {text_2}  
**First 5 values:** `{v2[:5]}`  
**Last 5 values:** `{v2[-5:]}`

---

### Vector Preview (Text 3)
**Text:** {text_3}  
**First 5 values:** `{v3[:5]}`  
**Last 5 values:** `{v3[-5:]}`
"""

display(Markdown(output))

# === Define Cosine Similarity Function ===
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector (list of floats)
        vec2: Second embedding vector (list of floats)
    
    Returns:
        Float between -1 and 1, where 1 means very similar
    """
    return 1 - cosine(vec1, vec2)

print("Cosine similarity function defined successfully!")

# === Calculate Similarity Scores ===
similarity_identical = cosine_similarity(v0, v1)  # Identical texts
similarity_related = cosine_similarity(v1, v2)    # Related texts
similarity_unrelated = cosine_similarity(v1, v3)  # Unrelated texts

# === Display Results ===
output = f"""
| Comparison | Cosine Similarity |
|------------|-------------------|
| **'{text_0}' vs '{text_1}'** (identical) | **{similarity_identical:.4f}** |
| '{text_1}' vs '{text_2}' (related) | {similarity_related:.4f} |
| '{text_1}' vs '{text_3}' (unrelated) | {similarity_unrelated:.4f} |
"""

display(Markdown(output))


# ============================================================
# GRANULARITY + TOPIC DILUTION COMPARISON
# ============================================================

# === User query and candidate texts ===
query = "How do I learn to code?"

word_text = "Python"

sentence_text = "Start learning to code by picking one language, practicing regularly, and building small projects."

focused_paragraph_text = """To learn coding, start with fundamentals like variables, loops, and functions.
Practice by writing small programs, then build simple projects to apply what you learn.
When you get stuck, read documentation and debug step-by-step; this is part of the process."""

mixed_paragraph_text = """Learning to code takes practice and patience. Start with the basics and build small projects.
Also, spend time polishing your resume, optimizing your LinkedIn profile, preparing for interviews,
and negotiating salary. A strong portfolio matters, but career strategy is important too."""

unrelated_paragraph_text = """A good workout plan balances strength training and cardio.
Track progress, prioritize recovery, and adjust your routine over time."""

# -------------------------
# Embed all texts in single API call
# -------------------------
texts = [query, word_text, sentence_text, focused_paragraph_text, mixed_paragraph_text, unrelated_paragraph_text]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

query_emb = response.data[0].embedding
word_emb = response.data[1].embedding
sentence_emb = response.data[2].embedding
focused_paragraph_emb = response.data[3].embedding
mixed_paragraph_emb = response.data[4].embedding
unrelated_paragraph_emb = response.data[5].embedding


# -------------------------
# Calculate similarities
# -------------------------
word_sim = cosine_similarity(query_emb, word_emb)
sentence_sim = cosine_similarity(query_emb, sentence_emb)
focused_paragraph_sim = cosine_similarity(query_emb, focused_paragraph_emb)
mixed_paragraph_sim = cosine_similarity(query_emb, mixed_paragraph_emb)
unrelated_paragraph_sim = cosine_similarity(query_emb, unrelated_paragraph_emb)

# -------------------------
# Helper function for text preview
# -------------------------
def preview(text: str, n: int = 110) -> str:
    t = text.replace("\n", " ").strip()
    return t if len(t) <= n else t[:n] + "..."

# -------------------------
# Display results
# -------------------------
output = f"""
## GRANULARITY: Word vs Sentence vs Paragraph(s)

**Query:** "{query}"

---

### Dimensionality Check

All texts are represented in the same embedding space:

| Text Type | Dimensions |
|-----------|------------|
| Query | {len(query_emb)} |
| Word | {len(word_emb)} |
| Sentence | {len(sentence_emb)} |
| Focused paragraph | {len(focused_paragraph_emb)} |
| Mixed paragraph | {len(mixed_paragraph_emb)} |
| Unrelated paragraph | {len(unrelated_paragraph_emb)} |

---

### Similarity Scores (against query "{query}")

| Text Type | Word Count | Text Preview | Similarity |
|-----------|------------|--------------|------------|
| **Word** | {len(word_text.split())} | "{word_text}" | {word_sim:.4f} |
| **Sentence** | {len(sentence_text.split())} | "{sentence_text}" | {sentence_sim:.4f} |
| **Focused paragraph** | {len(focused_paragraph_text.split())} | "{preview(focused_paragraph_text)}" | {focused_paragraph_sim:.4f} |
| **Mixed paragraph** | {len(mixed_paragraph_text.split())} | "{preview(mixed_paragraph_text)}" | {mixed_paragraph_sim:.4f} |
| **Unrelated paragraph** | {len(unrelated_paragraph_text.split())} | "{preview(unrelated_paragraph_text)}" | {unrelated_paragraph_sim:.4f} |
"""

display(Markdown(output))





