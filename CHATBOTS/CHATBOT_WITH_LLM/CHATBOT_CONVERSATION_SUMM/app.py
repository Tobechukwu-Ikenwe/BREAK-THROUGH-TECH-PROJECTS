# This app adds automatic conversation summarization to manage long conversations.
# Complete the code to enable summarization.

# Instructions:
# 1. Read through the guide
# 2. Find the BEGIN SOLUTION / END SOLUTION blocks below and complete it. DO NOT remove 
#    the BEGIN SOLUTION / END SOLUTION comments
# 3. Run 'streamlit run app.py' in the terminal
# 4. Test your chatbot in the browser
# 5. Check your code in the guide

import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="eCornell AI Chatbot",
    page_icon="🤖",
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
        "<h3 style='color: #b31b1b; margin-bottom: 0;'>🤖 My AI Chatbot with Smart Context Management</h3>",
        unsafe_allow_html=True,
    )
    st.caption("Powered by eCornell")

st.markdown("---")

# ============================================
# HELPER FUNCTIONS (PROVIDED)
# ============================================

def summarize_conversation(messages: list[dict]) -> tuple[str, int]:
    """
    Summarize the entire conversation using OpenAI's API.
    Returns the summary text and total tokens used for this API call.

    Uses gpt-4o-mini (a smaller, faster model) instead of gpt-4o because:
    - Summarization is a simpler task that doesn't need the most powerful model
    - gpt-4o-mini is much cheaper (~1/15th the cost) and faster
    - It's still excellent at summarization tasks

    This is a common pattern: use powerful models for user-facing responses,
    and smaller models for background tasks like summarization.
    """
    conversation_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
    summary_prompt = f"Summarize the following conversation concisely:\n{conversation_text}\nSummary:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Smaller, cheaper model for summarization
        messages=[{"role": "user", "content": summary_prompt}],
        # BEGIN SOLUTION - maximum output length
        max_completion_tokens = 750 
        # END SOLUTION - maximum output length
    )

    # Get token usage from API response
    summarization_tokens = response.usage.prompt_tokens + response.usage.completion_tokens

    return response.choices[0].message.content, summarization_tokens

# ============================================
# INITIALIZE SESSION STATE
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "last_input_tokens" not in st.session_state:
    st.session_state.last_input_tokens = 0

if "last_output_tokens" not in st.session_state:
    st.session_state.last_output_tokens = 0

if "last_summarization_tokens" not in st.session_state:
    st.session_state.last_summarization_tokens = 0

# ============================================
# SIDEBAR: SUMMARY AND TOKEN TRACKING
# ============================================

with st.sidebar:
    st.header("📊 Conversation Summary")
    if st.session_state.summary:
        st.write(st.session_state.summary)
    else:
        st.info("Summary will appear here after a few messages.")

    st.header("💰 Token Usage")

    # Show last interaction breakdown if available
    if st.session_state.last_input_tokens > 0:
        st.markdown("**Last Chat Interaction:**")
        st.markdown(f"- Chat Input (sent to API): {st.session_state.last_input_tokens} tokens")
        st.markdown(f"- Chat Output (from API): {st.session_state.last_output_tokens} tokens")
        st.markdown(f"- Chat Total: {st.session_state.last_input_tokens + st.session_state.last_output_tokens} tokens")

        # Show summarization tokens if summarization happened
        if st.session_state.last_summarization_tokens > 0:
            st.markdown(f"- **Summarization API Call:** {st.session_state.last_summarization_tokens} tokens")
            st.caption("(Summarization uses gpt-4o-mini, which is much cheaper)")

        st.divider()

    st.markdown(f"**Cumulative Total:** {st.session_state.total_tokens} tokens")
    st.caption("Lower tokens = lower costs!")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
        st.session_state.total_tokens = 0
        st.session_state.summary = ""
        st.session_state.last_input_tokens = 0
        st.session_state.last_output_tokens = 0
        st.session_state.last_summarization_tokens = 0
        st.rerun()

# ============================================
# DISPLAY CONVERSATION HISTORY
# ============================================

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])
    elif msg["role"] == "system":
        # System messages (summaries) are displayed differently
        st.info(f"🔄 {msg['content']}")

# ============================================
# CHAT INPUT & AUTOMATIC SUMMARIZATION
# ============================================

if prompt := st.chat_input("Type your message here..."):

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Reset summarization tokens for this turn
    st.session_state.last_summarization_tokens = 0

    # ============================================
    # GENERATE STREAMING AI RESPONSE
    # ============================================

    # Generate streaming AI response and collect token usage
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=st.session_state.messages,
        stream=True,
        stream_options={"include_usage": True}  # Get token usage from streaming response
    )

    # Collect the complete response and token usage
    full_response = ""
    input_tokens = 0
    output_tokens = 0

    for part in stream:
        # Collect text content as it arrives
        if part.choices and part.choices[0].delta.content:
            full_response += part.choices[0].delta.content

        # Collect usage data when available
        if hasattr(part, 'usage') and part.usage:
            input_tokens = part.usage.prompt_tokens
            output_tokens = part.usage.completion_tokens

    # Display the full response
    st.chat_message("assistant").write(full_response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Store tokens for this interaction (for sidebar display)
    st.session_state.last_input_tokens = input_tokens
    st.session_state.last_output_tokens = output_tokens
    st.session_state.total_tokens += input_tokens + output_tokens

    # ============================================
    # AUTOMATIC SUMMARIZATION (EVERY 3 USER MESSAGES)
    # ============================================

    # Count how many messages the user has sent
    user_message_count = sum(1 for msg in st.session_state.messages if msg['role'] == 'user')

    # Summarize every 3 user messages (after assistant has responded)
    if user_message_count > 0 and user_message_count % 3 == 0:

        # Pre-calculated variables (no complex indexing needed!)
        messages_to_summarize = st.session_state.messages[:-1]  # Everything except the latest
        current_assistant_message = st.session_state.messages[-1]     # Just the latest message

        # BEGIN SOLUTION - messages to summarize
        summary_text, summarization_tokens = summarize_conversation(messages_to_summarize) 
        # END SOLUTION - messages to summarize

        # Store the summary and token usage
        st.session_state["summary"] = summary_text
        st.session_state.last_summarization_tokens = summarization_tokens
        st.session_state.total_tokens += summarization_tokens

        # Replace long history with summary + current message
        st.session_state.messages = [
            {"role": "system", "content": f"This is a summary of the conversation so far: {st.session_state['summary']}"},
            current_assistant_message
        ]

    # Refresh to update sidebar
    st.rerun()

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
