import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def get_segmentation_results():
    """Load segmented data and generate segmentation insights."""
    df = pd.read_csv("uploads/user_segments.csv")

    # Count unique clusters
    cluster_counts = df['cluster_label'].value_counts().to_dict()

    # Generate a bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(cluster_counts.keys()), y=list(cluster_counts.values()))
    plt.xlabel("Segment")
    plt.ylabel("Count")
    plt.title("User Segmentation Distribution")

    # Save plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # Recommendations based on clusters
    recommendations = {
        "Browsers": "Send discount coupons to encourage engagement.",
        "Cart Abandoners": "Offer free delivery or discounts to reduce cart abandonment.",
        "Buyers": "Reward with loyalty points to encourage repeat purchases."
    }

    return cluster_counts, recommendations, plot_data
