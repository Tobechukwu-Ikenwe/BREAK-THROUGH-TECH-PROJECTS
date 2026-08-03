# This app enhances your chatbot with streaming responses.
# Complete the code to enable streaming.

# Instructions:
# 1. Read through the guide
# 2. Find the BEGIN SOLUTION / END SOLUTION block below and complete it. DO NOT remove 
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
    layout="centered"
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
        "<h3 style='color: #b31b1b; margin-bottom: 0;'>🤖 My Streaming AI Chatbot</h3>",
        unsafe_allow_html=True,
    )
    st.caption("Powered by eCornell")

st.markdown("---")

# ============================================
# INITIALIZE MESSAGE HISTORY
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# ============================================
# DISPLAY CONVERSATION HISTORY
# ============================================

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ============================================
# CHAT INPUT & STREAMING AI RESPONSE
# (This is the NEW part - streaming instead of waiting)
# ============================================

if prompt := st.chat_input("Type your message here..."):

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # ============================================
    # COMPLETE THIS STREAMING API CALL
    # ============================================
    # This is what's NEW!
    # You learned streaming concepts in the guide.

    # Create assistant chat container and stream the response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o",
            # BEGIN SOLUTION
            messages=st.session_state.messages,  
            stream = True  
            # END SOLUTION
        )

        # Display the stream in real-time and capture the full response
        full_response = st.write_stream(stream)

    # ============================================
    # SAVE AI RESPONSE TO HISTORY
    # ============================================

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

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
