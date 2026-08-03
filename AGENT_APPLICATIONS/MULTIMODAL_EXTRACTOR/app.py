import os
import json
import base64
import pandas as pd
from openai import OpenAI
from IPython.display import display, Markdown, Image as IPImage

# === OpenAI Client Setup ===
client = OpenAI()



def image_to_base64(file_path: str) -> str:
    """
    Reads an image file and returns its base64 encoded string.
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Path to the directory containing the receipt images
receipts_folder = os.path.join(os.getcwd(), "data", "receipts")

image_files = sorted(os.listdir(receipts_folder))

#no. of docs
print(f"Number of documents: {len(image_files)}")

# Do not remove or edit this cell
for filename in image_files:
    file_path = os.path.join(receipts_folder, filename)
    print(f"\n--- {filename} ---")
    display(IPImage(filename=file_path, width=400))


image_data = {}
for filename in image_files:
    file_path = os.path.join(receipts_folder, filename)
    image_data[filename] = image_to_base64(file_path)

extraction_prompt = """Extract the following fields from the receipt or invoice image into a valid JSON object:

1. vendor_name (string): The name of the vendor or business.
2. transaction_date (string): The date of the transaction in YYYY-MM-DD format.
3. items (list of objects): Each object must have:
   - description (string)
   - price (number)
4. tax_amount (number or null): The tax charged, if visible.
5. total_amount (number): The final total amount.
6. payment_method (string or null): How the bill was paid (e.g., "Credit Card", "Cash"), if visible.
7. spending_category (string): Classify into exactly one of these categories:
   - "Office Supplies"
   - "Food & Dining"
   - "Software & Technology"
   - "Travel"
   - "Professional Services"
   - "Other"

If a field is not present or visible in the image, set its value to null. Output only JSON."""



# Do not remove or edit this cell

# Get the first image
first_filename = image_files[0]
first_image_b64 = image_data[first_filename]

# Make the API call
test_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": extraction_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{first_image_b64}"}}
            ]
        }
    ],
    response_format={"type": "json_object"},
    max_tokens=1000
)

# Display the result
test_result = test_response.choices[0].message.content
print(f"Extraction result for {first_filename}:")
print("=" * 60)
print(test_result)


# Do not remove or edit this cell

parsed_results = {}

for filename, raw_json in raw_results.items():
    try:
        parsed = json.loads(raw_json)
        parsed_results[filename] = parsed
        print(f"Successfully parsed: {filename}")
    except json.JSONDecodeError as e:
        print(f"Error parsing {filename}: {e}")
        parsed_results[filename] = {"error": str(e)}

print(f"\nParsed {len(parsed_results)} documents.")



#analyze spending patterns
# Do not remove or edit this cell

all_extracted_data = json.dumps(parsed_results, indent=2)

print("Combined extracted data (preview):")
print(all_extracted_data[:500])
print("\n...")


# 1. System prompt defining the role as a financial analyst at SpendLens
analysis_system_prompt = """You are an expert financial analyst at SpendLens, a fintech company specializing in business expense automation. 
Your task is to analyze extracted expense data from multiple vendor documents and produce a clear, professional, and executive-ready summary report."""

# 2. User prompt containing the formatted extracted data and analytical tasks
analysis_user_prompt = f"""Please analyze the following extracted expense data across all submitted documents:

{all_extracted_data}

Provide a structured spending analysis that includes:
1. **Total Spending**: Calculate and report the total spending summed across all documents.
2. **Category Breakdown**: Provide a breakdown of spending by category, showing total expenditure and item count per category.
3. **Largest Expense**: Identify the single largest transaction, including vendor name, total amount, and category.
4. **Anomalies & Noteworthy Items**: Flag anything unusual or noteworthy (e.g., missing vendor/payment details, high-value transactions, or unclassified items)."""


# Do not remove or edit this cell

analysis_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": analysis_system_prompt},
        {"role": "user", "content": analysis_user_prompt}
    ],
    max_tokens=1500
)

analysis_text = analysis_response.choices[0].message.content

print("Spending Analysis")
print("=" * 60)
display(Markdown(analysis_text))



