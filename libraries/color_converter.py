import pandas as pd
import re

# Input and output file paths
input_file = 'sample-colors.txt'  
output_file = 'sample-colors.csv'

# Read the text file
with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Extract headers for the CSV file
headers = ["Color", "Name", "Hex Triplet", "Red", "Green", "Blue", "Hue", "Saturation", "Lightness", "Saturation (HSL)", "Value", "W3C Name"]

# Process each line into structured data
data = []
for line_num, line in enumerate(lines[2:]):  # Skip the first two lines (header and separator)
    line = line.strip()

    # Check if line is not empty
    if not line:
        print(f"Skipping empty line at line {line_num + 2}")
        continue

    # Remove excessive whitespace around the '|' character and split by '|'
    line = re.sub(r'\s*\|\s*', ' | ', line)  # Normalize whitespace around '|'
    parts = line.split(" | ")

    # Debugging: Print the number of parts in the line
    print(f"Line {line_num + 2} split into {len(parts)} parts: {parts}")

    # Check if the line has the correct number of columns
    if len(parts) != len(headers):  # If the line doesn't have the expected number of columns
        print(f"Skipping line at line {line_num + 2} due to incorrect number of columns: {line}")
        continue

    # Append the data row (leave W3C Name as is)
    data.append(parts)

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=headers)

# Save the data to a CSV file
df.to_csv(output_file, index=False)

print(f"File saved as {output_file}")
