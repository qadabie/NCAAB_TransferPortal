import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the parquet file
df = pd.read_parquet('weighted_jaccard_network_similarity.parquet')

# Select first 3 rows
sample_rows = df.sample(1, random_state=42)
print(sample_rows)

# Extract jaccard_similarities values
jaccard_values = sample_rows['jaccard_similarities'].tolist()
def create_similarity_plot(jaccard_values):
    """    Create a plot for the Jaccard similarity values.
    Args:
        jaccard_values (list): List of Jaccard similarity values.
    """
    # Create a figure
    fig = plt.figure(figsize=(10, 6))

    # Plot each row's jaccard similarities
    for i, values in enumerate(jaccard_values):
        plt.plot(values, marker='o', label=f'Row {i+1}')

    plt.title('Jaccard Similarities for First 3 Rows')
    plt.xlabel('Index')
    plt.ylabel('Jaccard Similarity Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a bit more style
    sns.set_style("whitegrid")
    plt.tight_layout()

    # Return the figure object
    return fig
