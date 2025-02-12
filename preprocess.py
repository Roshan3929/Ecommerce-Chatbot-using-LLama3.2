import pandas as pd 
import re

# Load your dataset
file_path = "cust_serv_know_base.csv"  
data = pd.read_csv(file_path)

#print(data.head())

# Function to preprocess text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters (if needed)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# Apply cleaning to the 'question' and 'answer' columns
data['Question'] = data['Question'].apply(clean_text)
data['Answer'] = data['Answer'].apply(clean_text)

data = data.dropna()

#print(data.head())

# Save the cleaned dataset to a new CSV file
output_file = "preprocessed_customer_service_data.csv"
data.to_csv(output_file, index=False)

print(f"Preprocessed data saved to {output_file}")
