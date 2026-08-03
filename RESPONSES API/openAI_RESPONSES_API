# === Imports ===
from openai import OpenAI
from IPython.display import display, Markdown

# === OpenAI Client Setup ===
client = OpenAI()

# YOUR CODE HERE
user_input = "who are the current premier league champions"
# END OF YOUR CODE

response = client.responses.create(
    model="gpt-4o",
    input=user_input
)


"""
As of the 2022-2023 season, Manchester City are the current Premier League champions.
This is a response based on the time of the models last training. 
"""

# Without instructions - just basic input
response_no_instructions = client.responses.create(
    model="gpt-4o",
    input="Explain what a neural network is."
)

print("WITHOUT Instructions:")
display(Markdown(response_no_instructions.output_text))
print("\n" + "="*80 + "\n")

# With instructions - guiding the tone and approach
response_with_instructions = client.responses.create(
    model="gpt-4o",
    instructions="You are a kindergarten teacher. Explain concepts using simple analogies that a 5-year-old would understand. Use playful language and emojis.",
    input="Explain what a neural network is."
)

print("WITH Instructions:")
display(Markdown(response_with_instructions.output_text))



# Display the response
display(Markdown(response.output_text))


# YOUR CODE HERE
topic = "explain keyboard driver support in operating systems"
first_instructions = "u are a university professor"
second_instructions = "u are explaining to a child"
# END OF YOUR CODE

# First response with first instructions
response1 = client.responses.create(
    model="gpt-4o",
    instructions=first_instructions,
    input=topic
)

print("First Response:")
display(Markdown(response1.output_text))
print("\n" + "="*80 + "\n")

# Second response with second instructions
response2 = client.responses.create(
    model="gpt-4o",
    instructions=second_instructions,
    input=topic
)

print("Second Response:")
display(Markdown(response2.output_text))


#Stateful Conversations

# First turn: introduce yourself
first_response = client.responses.create(
    model="gpt-4o",
    input="Hi, my name is Alex and I love mountain climbing."
)

print("First Response:")
display(Markdown(first_response.output_text))

# Save the response ID - this is key!
print(f"\nResponse ID: {first_response.id}")

# Second turn: ask about what you shared
second_response = client.responses.create(
    model="gpt-4o",
    input="What's my name and what hobby did I mention?",
    previous_response_id=first_response.id  # This is where we include the previous conversation ID.
)

print("Second Response:")
display(Markdown(second_response.output_text))

# Save the response ID - this is key!
print(f"\nResponse ID: {second_response.id}")


#Tool Usage

"""
We saw earlier that the model gave us a response based off the time of its last training.
Let us ask a different question.
"""

# Ask about a recent event
response_no_tool = client.responses.create(
    model="gpt-4o",
    input="Who won the 2024 Super Bowl?"
)

print("WITHOUT Web Search:")
display(Markdown(response_no_tool.output_text))

# Same question, but with web search enabled
response_with_tool = client.responses.create(
    model="gpt-4o",
    input="Who won the 2024 Super Bowl?",
    tools=[{"type": "web_search"}] # This is the tool call
)

print("WITH Web Search:")
display(Markdown(response_with_tool.output_text))

"""
Back to our Current Question
"""

# YOUR CODE HERE
current_info_question = "who are the current premier leagu champions."
# END OF YOUR CODE

response = client.responses.create(
    model="gpt-4o",
    input=current_info_question,
    tools=[{"type": "web_search"}]
)

display(Markdown(response.output_text))

"""
Now it outputs: Here are the current (2025‑26) Premier League champions:

Arsenal secured the 2025‑26 English Premier League title—marking their first championship in 22 years, since their famed “Invincibles” campaign in 2003‑04 (premierleague.com).
The title was officially confirmed on Tuesday, 19 May 2026, when Manchester City drew 1–1 with Bournemouth, making it mathematically impossible for them to overtake Arsenal in the standings (premierleague.com).

It called the web search tool 
"""

#File Search: Querying Your Own Documents

file_path = "data/Goddard_Paper_2019.pdf"

# Upload the file
file_response = client.files.create(
    file=open(file_path, "rb"),
    purpose="assistants"
)

print(f"File uploaded successfully! File ID: {file_response.id}")

# Create a vector store to uploaded file
vector_store = client.vector_stores.create(name="Research Papers")
print(f"Vector store created! ID: {vector_store.id}")

# Attach the file to the vector store
vs_file = client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file_response.id
)

print(f"File attached to vector store. Vector store file ID: {vs_file.id}")



"""

this is how vector stores look. my model splits the file to chunks and store them with the embeddings in vector store
| ID | Text | Embedding | Metadata |
|----|------|-----------|----------|
| 1 | "Python is a programming language." | `[0.23, -0.91, 0.44, ...]` | `{{"page": 1}}` |
| 2 | "Lists store multiple values." | `[0.51, 0.18, -0.72, ...]` | `{{"page": 2}}` |
"""

#QUERY

document_question = "How effective were the films at slowing ascorbic acid degradation?"

# Query the document using file search
response = client.responses.create(
    model="gpt-4o",
    input=document_question,
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store.id]
    }]
)

display(Markdown(response.output_text))

#CODE INTERPRETER TOOL

# Enable code interpreter tool
response = client.responses.create(
    model="gpt-4.1",
    instructions="You are a personal math tutor. When asked a math question, write and run code using the python tool to answer the question.",
    input="I need to calculate the compound interest on $5000 invested at 6% annual interest rate, compounded monthly, for 10 years. What will be the final amount?",
    tools=[{
        "type": "code_interpreter",
        
        "container": {
            "type": "auto"
        }
    }],
    include=["code_interpreter_call.outputs"]
)

# Show whether a tool was called
tool_called = any(getattr(i, "type", "").endswith("_call") for i in response.output)
if not tool_called:
    display(Markdown("## Tool called\n_No tool was called._"))

# Show tool call details (if any)
for item in response.output:
    if getattr(item, "type", "") == "code_interpreter_call":
        display(Markdown("## Tool called"))
        display(Markdown(f"- `{item.type}` (status: `{getattr(item, 'status', '')}`)"))

        display(Markdown("## Code"))
        display(Markdown(f"```python\n{item.code}\n```"))

        display(Markdown("## Tool output"))
        for out in (getattr(item, "outputs", None) or []):
            if getattr(out, "type", "") == "logs":
                display(Markdown(f"```text\n{out.logs}\n```"))
            else:
                display(Markdown(f"```text\n{out}\n```"))

display(Markdown("## Final output"))
display(Markdown(response.output_text))


"""
Tool called¶
code_interpreter_call (status: completed)
Code
# Given values
P = 5000         # Principal amount
r = 0.06         # Annual interest rate
n = 12           # Compounded monthly
t = 10           # Number of years

# Compound interest formula
A = P * (1 + r/n) ** (n*t)
A
Tool output
9096.98367016145
"""

#Putting It All Together: Tools + Conversation State

# First turn: Ask about the document
turn1 = client.responses.create(
    model="gpt-4o",
    input="What are the key applications mentioned in the uploaded research paper?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store.id]
    }]
)

print("Turn 1 (File Search):")
display(Markdown(turn1.output_text))
print("\n" + "="*80 + "\n")

# Second turn: Follow up with a web search (referencing the previous response)
turn2 = client.responses.create(
    model="gpt-4o",
    input="Are there any recent companies or startups working on these applications?",
    previous_response_id=turn1.id,  # Maintains conversation context
    tools=[{"type": "web_search"}]  # Now uses web search
)

print("Turn 2 (Web Search, maintaining context):")
display(Markdown(turn2.output_text))


"""
second response understood what was asked in the first because we passed in the previous id.
"""


#CUSTOM FUNCTION

import json

# ------------------------------------------------------------
# Step 1) Define the Python function we want the model to use.
# ------------------------------------------------------------
def get_current_weather(location, unit="fahrenheit"):
    """
    A mock weather function for demonstration.

    In a real application, you'd call a weather API here (OpenWeather, WeatherKit, etc.)
    and return real data. We'll return fake data so we can focus on the tool-calling flow.
    """
    weather_data = {
        "location": location,
        "temperature": "72" if unit == "fahrenheit" else "22",
        "unit": unit,
        "forecast": ["sunny", "windy"]
    }

    # Tools can return strings or structured data.
    # Returning JSON as a string is a common pattern.
    return json.dumps(weather_data)

# -------------------------------------------------------------------
# Step 2) Describe the function to the model using a JSON schema.
# -------------------------------------------------------------------
# This "tools" definition tells the model:
# - the tool name it can request ("get_current_weather")
# - what it does (description)
# - what arguments it accepts (parameters schema)
#
tools = [{
    "type": "function",
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use"
            }
        },
        "required": ["location"]
    }
}]

# -------------------------------------------------------------------
# Step 3) Ask the model a question, while enabling the custom tool.
# -------------------------------------------------------------------
response = client.responses.create(
    model="gpt-4o",
    input="What's the weather like in Boston?",
    tools=tools
)

# -------------------------------------------------------------------
# Step 4) Check whether the model requested a tool call.
# -------------------------------------------------------------------
# The Responses API returns a list of "output items" in response.output.
# Some items are normal assistant messages, and some items are tool calls.
#
# If the model wants to use our function, you'll see an item with:
#   type == "function_call"
# and fields like:
#   name      -> function name (e.g. "get_current_weather")
#   arguments -> JSON string of arguments (e.g. {"location":"Boston, MA"})
#   call_id   -> used to match this call to the tool output we send back
function_call = None
for item in response.output:
    if getattr(item, "type", "") == "function_call" and getattr(item, "name", "") == "get_current_weather":
        function_call = item
        break

# -------------------------------------------------------------------
# Step 5) If there was a tool call, run the function locally,
#         then send the tool result back to the model.
# -------------------------------------------------------------------
if function_call:
    # Parse the JSON arguments the model provided
    args = json.loads(function_call.arguments)

    # Execute the function in Python using those arguments
    function_result = get_current_weather(**args)

    # Show students that a tool was called and with which arguments
    display(Markdown("## Tool called?"))
    display(Markdown("**True**"))
    display(Markdown(f"- `{function_call.name}` with `{args}`"))

    # Show students what the tool returned
    display(Markdown("## Tool output"))
    display(Markdown(f"```json\n{function_result}\n```"))

    # Now we pass the tool result back to the model.
    # The model uses it to write the final natural-language response.
    response2 = client.responses.create(
        model="gpt-4o",
        previous_response_id=response.id,  # keeps the conversation context
        input=[{
            "type": "function_call_output",
            "call_id": function_call.call_id,  # must match the original function_call
            "output": function_result           # what our function returned
        }],
        tools=tools
    )

    # Final response (now grounded in the tool output)
    display(Markdown("## Final response"))
    display(Markdown(response2.output_text))

# -------------------------------------------------------------------
# Step 6) If there was no tool call, the model answered directly.
# -------------------------------------------------------------------
else:
    display(Markdown("## Tool called?"))
    display(Markdown("**False**"))

    # In this case, response.output_text should contain the final answer
    display(Markdown("## Final response"))
    display(Markdown(response.output_text))


                  
