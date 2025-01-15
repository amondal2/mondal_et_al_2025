import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.style.use("ggplot")
plt.rcParams["pdf.fonttype"] = 42

if __name__ == "__main__":
    df = pd.read_csv(
        os.path.expanduser("~/EMOD-calibration/emulator/multisite/sensitivity_df.csv")
    )
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    ax = sns.lineplot(x="frac", y="loss", data=df, errorbar="sd", color="#E69F00", lw=2)
    ax.grid(True)
    ax.margins(x=0, y=0)
    ax.set(
        xlabel="Dataset fraction",
        ylabel="Test set loss (MSE)",
    )
    ax.set_title("Emulator sensitivity (n=50 repetitions)", fontweight="bold")
    plt.savefig(
        os.path.expanduser("~/EMOD-calibration/emulator/multisite/sensitivity.pdf")
    )
