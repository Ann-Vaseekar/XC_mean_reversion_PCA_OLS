
import matplotlib.pyplot as plt


def plot_corr(corr):

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im)

    # Set ticks and labels
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)

    plt.title('Correlation Matrix of Stock Returns')
    plt.tight_layout()
    plt.show()

    # Calculate average correlation
    avg_correlation = (corr.sum().sum() - len(corr)) / (len(corr) * (len(corr) - 1))
    print(f"Average correlation between stocks: {avg_correlation:.3f}")