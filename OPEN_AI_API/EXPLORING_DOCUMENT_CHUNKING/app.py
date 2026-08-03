# === Import required libraries ===
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from IPython.display import display, Markdown
import json


# === Load the pharmaceutical data we'll use for experiments ===
loader = TextLoader("data/RAG_source.txt")
documents = loader.load()

output = f"""
## Source Document Loaded

**File:** {documents[0].metadata['source']}

**Total length:** {len(documents[0].page_content):,} characters

**Content preview:**

```
{documents[0].page_content}...
```
"""

display(Markdown(output))


# === Configuration 1: Balanced approach ===
# This is a common starting point for many RAG applications.
# The goal is to create chunks that are large enough to keep useful context, but small enough to retrieve selectively.

# --- Parameters (define once so we don't repeat hardcoded values) ---
chunk_size = 400
chunk_overlap = 50

# Create the splitter.
# - chunk_size sets the target maximum characters per chunk.
# - chunk_overlap sets how many characters should be shared between adjacent chunks.
splitter_balanced = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# Split the input documents (a list of LangChain Document objects) into smaller Documents ("chunks").
# Each chunk is stored as a Document with its text in `.page_content`.
# Invoking the function is simple, and we use the dot(.) operator for this.
chunks_balanced = splitter_balanced.split_documents(documents)

# Compute summary statistics for the chunks we created.
total_chunks = len(chunks_balanced)
avg_chunk_size = sum(len(c.page_content) for c in chunks_balanced) / total_chunks

# Calculate overlap as a percentage of chunk_size for reporting.
# (This is a descriptive metric for displaying only, not something the splitter directly uses.)
overlap_percent = (chunk_overlap / chunk_size) * 100

# Note: we use the variables above so the report stays correct if we change the parameters later.
output = f"""
## Configuration: Balanced Chunking

**Parameters:**
- `chunk_size`: {chunk_size} characters
- `chunk_overlap`: {chunk_overlap} characters ({overlap_percent:.1f}% overlap)

**Results:**
- **Total chunks created:** {total_chunks}
- **Average chunk size:** {avg_chunk_size:.1f} characters

**First chunk:**
{chunks_balanced[0].page_content}

**Second chunk:**
{chunks_balanced[1].page_content}

**Third chunk:**
{chunks_balanced[2].page_content}
"""

display(Markdown(output))



# === Create three different splitters ===

splitter_small = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40  # 20% overlap
)

splitter_medium = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100  # 20% overlap
)

splitter_large = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200  # 20% overlap
)

# === Split the document with each configuration ===
chunks_small = splitter_small.split_documents(documents)
chunks_medium = splitter_medium.split_documents(documents)
chunks_large = splitter_large.split_documents(documents)

# === Compare the results ===
output = f"""
## Chunk Size Comparison

| Configuration | Chunk Size | Overlap | Total Chunks | Avg Chunk Length |
|---------------|-----------|---------|--------------|------------|
| Small | {splitter_small._chunk_size} | {splitter_small._chunk_overlap} | {len(chunks_small)} | {sum(len(c.page_content) for c in chunks_small) / len(chunks_small):.1f} |
| Medium | {splitter_medium._chunk_size} | {splitter_medium._chunk_overlap} | {len(chunks_medium)} | {sum(len(c.page_content) for c in chunks_medium) / len(chunks_medium):.1f} |
| Large | {splitter_large._chunk_size} | {splitter_large._chunk_overlap} | {len(chunks_large)} | {sum(len(c.page_content) for c in chunks_large) / len(chunks_large):.1f} |

"""

display(Markdown(output))


# === Find chunks containing "Zelomax" for each configuration ===

def find_keyword_chunks(chunks, keyword: str):
    k = keyword.lower()
    return [c for c in chunks if k in (c.page_content or "").lower()]

small_zelomax  = find_keyword_chunks(chunks_small,  "Zelomax")
medium_zelomax = find_keyword_chunks(chunks_medium, "Zelomax")
large_zelomax  = find_keyword_chunks(chunks_large,  "Zelomax")

output = f"""
## How Different Chunk Sizes Handle the Same Information

**Question:** "What are the side effects and contraindications of Zelomax?"

Let's see how many chunks contain "Zelomax" in each configuration:

- **Small chunks:** {len(small_zelomax)} chunks mention Zelomax
- **Medium chunks:** {len(medium_zelomax)} chunks mention Zelomax
- **Large chunks:** {len(large_zelomax)} chunks mention Zelomax

### Small Chunks (200 characters)

**Zelomax chunk:**
```
{small_zelomax[1].page_content}
```

### Medium Chunks (500 characters)

**Zelomax chunk:**
```
{medium_zelomax[1].page_content}
```

### Large Chunks (1000 characters)

**Zelomax chunk:**
```
{large_zelomax[0].page_content if large_zelomax else 'No chunks found'}
```

"""

display(Markdown(output))



# === Create splitters with different overlap amounts ===

splitter_no_overlap = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

splitter_small_overlap = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10  # 10% overlap
)

splitter_medium_overlap = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=25  # 25% overlap
)

# === Split and analyze ===
chunks_no_overlap = splitter_no_overlap.split_documents(documents)
chunks_small_overlap = splitter_small_overlap.split_documents(documents)
chunks_medium_overlap = splitter_medium_overlap.split_documents(documents)


# Approximate "storage" by total characters stored across all chunks
total_chars_no = sum(len(c.page_content) for c in chunks_no_overlap)
total_chars_small = sum(len(c.page_content) for c in chunks_small_overlap)
total_chars_medium = sum(len(c.page_content) for c in chunks_medium_overlap)

output = f"""
## Overlap Comparison (Fixed Chunk Size: {splitter_no_overlap._chunk_size})

| Overlap Strategy | Overlap Amount | Total Chunks | Approx. Storage (chars) | Storage Efficiency vs No Overlap |
|------------------|----------------|--------------|--------------------------|----------------------------------|
| No Overlap | {splitter_no_overlap._chunk_overlap} | {len(chunks_no_overlap)} | {total_chars_no:,} | 100.0% |
| Small Overlap | {splitter_small_overlap._chunk_overlap} ({splitter_small_overlap._chunk_overlap / splitter_small_overlap._chunk_size * 100:.0f}%) | {len(chunks_small_overlap)} | {total_chars_small:,} | {total_chars_no / total_chars_small * 100:.1f}% |
| Medium Overlap | {splitter_medium_overlap._chunk_overlap} ({splitter_medium_overlap._chunk_overlap / splitter_medium_overlap._chunk_size * 100:.0f}%) | {len(chunks_medium_overlap)} | {total_chars_medium:,} | {total_chars_no / total_chars_medium * 100:.1f}% |
"""

display(Markdown(output))


# === Compare consecutive chunks with and without overlap ===

output = f"""
## Consecutive Chunks: With vs Without Overlap

### No Overlap Configuration

**Chunk 3 ends with:**
```
...{chunks_no_overlap[2].page_content[-100:]}
```

**Chunk 4 starts with:**
```
{chunks_no_overlap[3].page_content[:100]}...
```

### Medium Overlap Configuration (100 characters)

**Chunk 3 ends with:**
```
...{chunks_medium_overlap[2].page_content[-150:]}
```

**Chunk 4 starts with:**
```
{chunks_medium_overlap[3].page_content[:150]}...
```
"""

display(Markdown(output))

# === Check metadata on a chunk ===
sample_chunk = chunks_medium[5]

output = f"""
## Metadata Preservation Example

**Chunk content:**
```
{sample_chunk.page_content}
```

**Chunk metadata:**
```python
{sample_chunk.metadata}
```
"""

display(Markdown(output))


# === Test with a very short "document" ===
from langchain_core.documents import Document

short_doc = Document(
    page_content="This is a very short document with just one sentence.",
    metadata={"source": "test"}
)

chunks_from_short = splitter_medium.split_documents([short_doc])

output = f"""
## Edge Case: Very Short Document

**Original document length:** {len(short_doc.page_content)} characters

**Chunk size setting:** {splitter_medium._chunk_size}

**Number of chunks created:** {len(chunks_from_short)}
"""

display(Markdown(output))




