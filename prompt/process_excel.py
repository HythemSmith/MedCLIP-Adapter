import pandas as pd

# Read the Excel file
df = pd.read_excel('E:\\MedCLIP-Adapter\\MedCLIP-Adapter\\prompt\\dataset.xlsx')

# Get the column names for columns E to S
position_column_names = df.columns[4:19]

# Function to find the position name
def get_position(row):
    positions = []
    for col_name in position_column_names:
        if row[col_name] == 1:
            positions.append(col_name)
    return ','.join(positions)

# Apply the function to create the 'position' column
df['position'] = df.apply(get_position, axis=1)

# Select the "name" column (assuming it's the first column, column A)
df['name'] = df.iloc[:, 0]

# Create a new DataFrame with only the "name" and "position" columns
output_df = df[['name', 'position']]

# Write the output to a CSV file
output_df.to_csv('E:\\MedCLIP-Adapter\\MedCLIP-Adapter\\prompt\\output.csv', index=False)

print("Successfully created output.csv with new format")