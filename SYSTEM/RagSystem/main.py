# === Import required libraries ===

# Standard imports
import os
from IPython.display import display, Markdown

# OpenAI
import openai
from openai import OpenAI

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

# === Setup ===
client = OpenAI()


# === Load the pharmaceutical data ===
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


# === Split documents into chunks ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

output = f"""
## Text Splitting Complete

**Splitter:** RecursiveCharacterTextSplitter
**Chunk size:** {text_splitter._chunk_size} characters
**Chunk overlap:** {text_splitter._chunk_overlap} characters

**Results:**
- **Original documents:** {len(documents)}
- **Chunks created:** {len(chunks)}

**Sample chunk:**
```
{chunks[0].page_content}
```
"""

display(Markdown(output))


# === Create embeddings and vector store ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

output = f"""
## Vector Store Created

**Embedding model:** {embeddings.model}

**Chunks indexed:** {len(chunks)}
"""

display(Markdown(output))


# === Create a retriever from the vector store ===
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Test the retriever
test_query = "Is Zelomax safe during pregnancy?"
retrieved_docs = retriever.invoke(test_query)

output = f"""
## Retriever Setup

**Configuration:**
- **Search type:** {retriever.search_type}
- **Number of results (k):** {retriever.search_kwargs.get("k")}

**Test query:** "{test_query}"

**Retrieved chunks:**
"""

for i, doc in enumerate(retrieved_docs, 1):
    output += f"""
### Chunk {i}:
```
{doc.page_content}
```

# === Initialize the ChatOpenAI wrapper ===
llm = ChatOpenAI(model="gpt-4o")

# Test it directly
response = llm.invoke([HumanMessage(content="What is RAG in 1 sentence?")])

output = f"""
## ChatOpenAI Test

**Query:** "What is RAG in 1 sentence?"

**Response:**
{response.content}
"""

display(Markdown(output))

# === PromptTemplate with placeholders ===
template = """Use this context to answer the question.

Context:
{context}

Question:
{question}

Answer:

"""

# Create a PromptTemplate object from the template string
custom_prompt = PromptTemplate.from_template(template)

# === Example runtime inputs ===
example_inputs = {
    "context": "Zelomax is an experimental medication. Common side effects reported include nausea and dizziness.",
    "question": "What are two possible side effects of Zelomax?"
}

# === Render the final prompt (what the LLM actually receives) ===
rendered_prompt = custom_prompt.format(**example_inputs)

output = f"""
## PromptTemplate Example

### 1) The template (with placeholders)
```text
{template}
```

### 2) What input variables does LangChain expect?
```python
{custom_prompt.input_variables}
```

### 3) Example inputs you might pass at runtime
```python
{example_inputs}
```

### 4) The rendered prompt (placeholders filled in)
```text
{rendered_prompt}
```
"""

display(Markdown(output))

# YOUR CODE HERE
rag_instruction = """
use this format to answer
say idk if answer is not in context

Question: {question}
Context: {context}
Answer:

"""# END OF YOUR CODE

# Create the PromptTemplate
prompt = PromptTemplate.from_template(rag_instruction)

# Display your prompt
output = f"""
## Your RAG Prompt

**Template:**
```
{rag_instruction}
```

**Variables detected:** `{prompt.input_variables}`

"""

display(Markdown(output))



"""

display(Markdown(output))




