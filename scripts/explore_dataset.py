import pandas as pd

# Sample data
data = pd.read_csv('Datasets/all_files_ER_fulltexts_robotsenergy.csv')
df = pd.DataFrame(data)

# Get unique values for the 'type' column
unique_types = df['type'].unique()

print(unique_types)
# Output: ['A' 'B' 'C']

# Filter the DataFrame
filtered_df = df[df['type'] == 'CONSUMER_GOOD']

# Save the filtered data to a new CSV file instead of printing
filtered_df.to_csv('filtered_consumer_goods_robotsenergy.csv', index=False)
