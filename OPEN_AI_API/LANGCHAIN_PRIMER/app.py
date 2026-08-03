# === Document Loading ===
from langchain_community.document_loaders import TextLoader

# === Embeddings & Vector Store ===
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# === Display ===
from IPython.display import display, Markdown

print("✓ LangChain packages imported successfully!")


# === Load document from file ===
# TextLoader reads a text file and converts it to LangChain Document objects
# Each Document has:
#   - page_content: The actual text from the file
#   - metadata: Information about the source (file path, etc.)
loader = TextLoader("data/RAG_source.txt")

# === Execute the loading ===
# load() returns a list of Document objects (usually just one for text files)
documents = loader.load()

# === Display document info ===
output = f"""
## Document Loaded

**Number of documents:** {len(documents)}

**Metadata:**
```python
{documents[0].metadata}
```

**Content preview:**
```
{documents[0].page_content}...
```

**Total length:** {len(documents[0].page_content):,} characters
"""

display(Markdown(output))




# === Very simple chunking: Split into 5 equal parts ===

# Get the full document text
full_text = documents[0].page_content

# Calculate the size of each chunk (divide by 5)
total_length = len(full_text)
num_chunks = 5 # number of chunks
chunk_size = total_length // num_chunks

# Create 5 chunks
chunks = []
for i in range(num_chunks):
    start = i * chunk_size
    # For the last chunk, take everything remaining to avoid losing characters
    if i == num_chunks-1:
        end = total_length
    else:
        end = (i + 1) * chunk_size
    
    chunk_text = full_text[start:end]
    
    # Store as a simple dict (dictionary)
    chunks.append({
        "content": chunk_text,
        "chunk_number": i + 1,
        "start_position": start,
        "end_position": end
    })

# Display results
output = f"""
## Chunking Results

**Original document length:** {len(full_text):,} characters

**Number of chunks:** {len(chunks)} (split into equal parts)

**Chunk sizes:**
- Chunk 1: {len(chunks[0]['content'])} characters
- Chunk 2: {len(chunks[1]['content'])} characters
- Chunk 3: {len(chunks[2]['content'])} characters
- Chunk 4: {len(chunks[3]['content'])} characters
- Chunk 5: {len(chunks[4]['content'])} characters

**Text from each chunk:**

**Chunk 1:**
```
{chunks[0]['content']}
```

**Chunk 2:**
```
{chunks[1]['content']}
```

**Chunk 3:**
```
{chunks[2]['content']}
```

**Chunk 4:**
```
{chunks[3]['content']}
```

**Chunk 5:**
```
{chunks[4]['content']}
```

"""

display(Markdown(output))



# === Create embeddings and index into ChromaDB ===
# Goal: Take our chunks, embed them into vectors, and store them in a vector database (Chroma)

# -----------------------------
# Step 1: Convert chunks -> LangChain Document objects
# -----------------------------
# Many vector stores supported by LangChain expect a list of Document objects.
# Each Document has:
#   - page_content: the actual text to embed
#   - metadata: any extra info you want to store alongside that text
# We will define these explicitly below

from langchain_core.documents import Document

print("Step 1: Converting chunks to Document objects...")

chunk_documents = []

# We'll store a source label in metadata if it exists; otherwise fall back to "unknown".
# This is useful later when you want to trace a retrieved chunk back to its origin.
source_value = documents[0].metadata.get("source", "unknown") if documents else "unknown"

for i, chunk in enumerate(chunks):
    doc = Document(
        page_content=chunk["content"],
        metadata={
            # Where did this text come from? (file path, URL, etc.)
            "source": source_value,

            # A simple identifier we assign ourselves, based on the for loop
            "chunk_id": i,

            # Optional: extra fields we tracked during chunking
            "chunk_number": chunk["chunk_number"],
            "start_position": chunk["start_position"],
            "end_position": chunk["end_position"],
        },
    )
    chunk_documents.append(doc)

print(f"  ✓ Created {len(chunk_documents)} Document objects")


# -----------------------------
# Step 2: Initialize an embedding model call
# -----------------------------
# This model converts text into a fixed-length numeric vector.
# Note: The embedding dimension depends on the model you choose.
# - text-embedding-3-small: 1,536 dimensions (we will use this for the example here)
# - text-embedding-3-large: 3,072 dimensions

print("\nStep 2: Initializing embedding model...")

embedding_model_name = "text-embedding-3-small" 
embedding_dims = 1536

import os
print("OPENAI_API_KEY visible?", bool(os.getenv("OPENAI_API_KEY")))
print("First 8 chars:", (os.getenv("OPENAI_API_KEY") or "")[:8])

embeddings = OpenAIEmbeddings(model=embedding_model_name)
print(f"  ✓ Using {embedding_model_name} ({embedding_dims:,} dimensions)")


# -----------------------------
# Step 3: Create embeddings and index into ChromaDB
# -----------------------------
# This single call does a lot:
#   1) Reads each Document.page_content
#   2) Calls the embedding model to create vectors
#   3) Stores vectors + original text + metadata in ChromaDB
#   4) Prepares the index for fast similarity search

print("\nStep 3: Creating embeddings and indexing into ChromaDB...")
print("  (This may take a moment as we call the embedding API...)")

vectorstore = Chroma.from_documents(
    documents=chunk_documents,
    embedding=embeddings,
)

print(f"  ✓ Successfully indexed {len(chunk_documents)} chunks")


# -----------------------------
# Step 4: Inspect what we stored
# -----------------------------
# We'll run a tiny similarity search against a user query just to pull back one stored chunk to show what a retrieved Document looks like.
# Remember that top-k determines how many chunks are retrieved. Here we will retrieved only one chunk, so k=1.

user_query = "Is Zelomax safe during pregnancy?"

print("\nStep 4: Inspecting the vector store...")

sample_results = vectorstore.similarity_search(user_query, k=1)
sample_doc = sample_results[0] if sample_results else None

# Accessing the list of vectors (floating point numbers) for a specific embedding
vector_preview = vectorstore._collection.get(include=["embeddings"], limit=1)["embeddings"][0][:10]

output = f"""
## Vector Store Created Successfully!

### Storage Summary

| Metric | Value |
|--------|-------|
| Chunks indexed | {len(chunk_documents)} |
| Embedding dimensions | {embedding_dims:,} |
| Total vectors stored | {len(chunk_documents)} |
| Vector database | ChromaDB (in-memory) |

### Example of Retrieved Chunk

**User Query:**
{user_query}

**Chunk Metadata:**

{sample_doc.metadata if sample_doc else 'N/A'}

**Chunk Content preview:**

{sample_doc.page_content if sample_doc else 'N/A'}

**Vector preview (first 10 dimensions):**

{vector_preview}

"""

display(Markdown(output))


# === Build the Retrieval Chain ===

from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Step 1: Create a retriever from our vector store
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": len(chunk_documents)}  # Get all chunks for this demo
)

# Step 2: Create a formatting function
def format_docs(docs):
    """Takes a list of Documents and formats them into a single string, separated by ---."""
    return "\n\n---\n\n".join([
        f"\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])

# Step 3: Build the retrieval chain
# Note: We wrap in RunnableParallel to make the dictionary invokable directly.
# When piped to other components (like | prompt | llm), the wrapper isn't needed.
retrieval_chain = RunnableParallel({
    "context": retriever | format_docs,      # Retrieve and format
    "question": RunnablePassthrough()        # Pass through unchanged
})

# Step 4: Execute the chain
query = "What are the side effects of Zelomax?"
result = retrieval_chain.invoke(query)

# Display results
output = f"""
## Retrieval Chain Results

**Query:** "{query}"

**Output structure:**
```python
{{
    "context": "<formatted chunks>",
    "question": "<original query>"
}}
```

**Retrieved Context:**
```
{result['context']}...
```

**Preserved Question:**
```
{result['question']}
```
"""

display(Markdown(output))



  

