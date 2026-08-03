# This app implements retrieval-augmented generation (RAG), a powerful technique for enabling
# LLMs to answer questions based on specific knowledge bases without exceeding context window limits.
# You'll learn how to: Retrieve relevant documents, Augment prompts with that content, and Generate informed responses.
# Complete the code by writing effective system prompts for each stage of RAG.

# Instructions:
# 1. Read through the guide
# 2. Find the TWO BEGIN SOLUTION / END SOLUTION blocks below and complete them
# 3. Run 'streamlit run app.py' in the terminal
# 4. Test your chatbot in the browser by asking questions about Harvard, Cornell, or Duke!

import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="eCornell RAG Chatbot",
    page_icon="🔍",
    layout="wide"
)

# ============================================
# CORNELL HEADER
# ============================================

col1, col2 = st.columns([1, 4])
with col1:
    st.image(
        "cornell_seal.png",
        width=100,
    )
with col2:
    st.markdown(
        "<h3 style='color: #b31b1b; margin-bottom: 0;'>🔍 My First RAG Chatbot</h3>",
        unsafe_allow_html=True,
    )
    st.caption("Powered by eCornell")

st.markdown("---")

# ============================================
# INITIALIZE SESSION STATE
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have knowledge about Harvard, Cornell, and Duke universities. Ask me anything about these schools!"}
    ]

if "last_file" not in st.session_state:
    st.session_state.last_file = "none"

# ============================================
# HELPER FUNCTION: RETRIEVE RELEVANT KNOWLEDGE BASE
# ============================================

def get_knowledge_file(prompt: str) -> str:
    """
    Uses an LLM to classify which knowledge base file is most relevant to the user's question.
    This is the RETRIEVAL step in RAG - finding the right information before generating a response.

    Returns the filename (e.g., "harvard.txt", "cornell.txt", "duke.txt", or "none.txt")
    """
    # Your task: Replace the YOUR CODE HERE with your classification prompt that:
    # 1. Explains the task (determine which knowledge base file is relevant)
    # 2. Lists all available options: harvard.txt, cornell.txt, duke.txt, none.txt
    # 3. Specifies the output format (return ONLY the filename)
    #
    # Hint: Be clear about what each file contains and when to use "none.txt"

    response = client.chat.completions.create(
        model="gpt-4o-mini",    
        messages=[
            # BEGIN SOLUTION
            {"role": "system", 
    "content": """
You are a knowledge base classifier.

Your job is to determine which knowledge base file contains the information most relevant to the user's question.

Available knowledge bases:

1. harvard.txt
   - Contains information about Harvard University, including admissions, programs, campus, history, and related topics.

2. cornell.txt
   - Contains information about Cornell University, including admissions, programs, campus, history, and related topics.

3. duke.txt
   - Contains information about Duke University, including admissions, programs, campus, history, and related topics.

4. none.txt unrelated information

Classification rules:
- Choose the single file that is most relevant to the user's question.
- If the question mentions a university, choose the matching university file.
- If the question is about multiple universities, choose the file that is most likely to contain the answer.
- If the question is unrelated to these universities, choose none.txt.

Output requirements:
- Your response must contain ONLY the filename.
- Do not include explanations, punctuation, formatting, or additional text.

Valid outputs:
harvard.txt
cornell.txt
duke.txt
none.txt
"""},
            # END SOLUTION
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# ============================================
# SIDEBAR: KNOWLEDGE BASE INFO AND RESET
# ============================================

with st.sidebar:
    st.header("📚 Knowledge Base Info")

    # Create a dynamic container for displaying which knowledge base was last used
    # This allows us to update the display in real-time as files are selected
    sidebar_file_info = st.empty()

    # Initial display based on current session state
    if st.session_state.last_file != "none":
        sidebar_file_info.success(f"**Last used:** {st.session_state.last_file}")
    else:
        sidebar_file_info.info("**No knowledge base file retrieved for this API call**")

    st.caption("The chatbot retrieves relevant files to answer your questions")

    st.markdown("---")

    # Available knowledge bases
    st.subheader("Available Knowledge:")
    st.markdown("- 🎓 **harvard.txt** - Harvard University info")
    st.markdown("- 🎓 **cornell.txt** - Cornell University info")
    st.markdown("- 🎓 **duke.txt** - Duke University info")

    st.markdown("---")

    # Reset conversation button
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I have knowledge about Harvard, Cornell, and Duke universities. Ask me anything about these schools!"}
        ]
        st.session_state.last_file = "none"
        st.rerun()

# ============================================
# DISPLAY CONVERSATION HISTORY
# ============================================

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ============================================
# HANDLE USER INPUT AND RAG WORKFLOW
# ============================================

if prompt := st.chat_input("Ask a question about Harvard, Cornell, or Duke..."):

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # ============================================
    # STEP 1: RETRIEVE - Determine which knowledge base to use
    # ============================================

    # Call the helper function you wrote above to classify the question
    # This determines which file contains relevant information
    selected_file = get_knowledge_file(prompt)

    # ============================================
    # STEP 2 & 3: AUGMENT & GENERATE
    # ============================================

    # Check if a specific knowledge base file was selected (not "none.txt")
    if selected_file in ["harvard.txt", "cornell.txt", "duke.txt"]:
        try:
            # Construct the file path (files are in ../data/knowledge_base/ directory)
            file_path = f"data/knowledge_base/{selected_file}"

            # Read the content of the selected knowledge base file
            # This is the information we'll use to augment our prompt
            with open(file_path, "r") as file:
                content = file.read()

            # Update session state to remember which file was used
            st.session_state.last_file = selected_file

            # Update sidebar display immediately to show which file is being used
            sidebar_file_info.success(f"**Last used:** {selected_file}")

            # ============================================
            # AUGMENTED GENERATION 
            # ============================================

            # Now we have the relevant document content, we need to augment our prompt with it
            # and generate a response. This is the AUGMENT + GENERATE steps of RAG.

            with st.chat_message("assistant"):
                # Your task: Replace YOUR CODE HERE with your augmented generation prompt that:
                # 1. Instructs the LLM to answer based on the provided document content
                # 2. Includes the 'content' variable using f-string: {content}
                # 3. Tells the LLM what to do if the answer isn't in the content
                #
                # Hint: Use f"""...""" to create an f-string that includes {content}

                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        # BEGIN SOLUTION
                        {"role": "system", "content": f"""{content}    Instructions:
- Base your answer on the provided document.
- Do not rely on outside knowledge if the information is not contained in the document.
- If the answer cannot be found in the provided content, clearly state that the information is not available.
- Provide a helpful and informative response. """},                
                        # END SOLUTION
                        *st.session_state.messages  # The * unpacks conversation history to maintain context
                    ],
                    stream=True
                )
                response_text = st.write_stream(stream)

                # Add the response to conversation history
                st.session_state.messages.append({"role": "assistant", "content": response_text})

        except FileNotFoundError:
            # Handle case where the file doesn't exist
            st.error(f"Error: Knowledge base file '{selected_file}' not found.")
            response_text = "I apologize, but I couldn't access the information needed to answer your question."
            st.session_state.last_file = "none"
            sidebar_file_info.info("**No knowledge base file retrieved for this API call**")
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            # Handle other potential errors
            st.error(f"An unexpected error occurred: {e}")
            response_text = "Sorry, I encountered an error while processing your request."
            st.session_state.last_file = "none"
            sidebar_file_info.info("**No knowledge base file retrieved for this API call**")
            st.session_state.messages.append({"role": "assistant", "content": response_text})

    else:
        # ============================================
        # STANDARD CHAT (No RAG) - When no relevant knowledge base is found
        # ============================================

        st.session_state.last_file = "none"
        sidebar_file_info.info("**No knowledge base file retrieved for this API call**")

        # Fall back to standard chatbot response without any document context
        # The LLM will answer based on its general training knowledge
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state.messages,
                stream=True
            )
            response_text = st.write_stream(stream)

        # Add the response to conversation history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© eCornell<br>"
    "For assistance, contact course staff"
    "</div>",
    unsafe_allow_html=True,
)
