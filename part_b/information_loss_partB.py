import matplotlib.pyplot as plt
import pandas as pd

checkpoint_data = {
    "Epoch": [1, 5, 10, 20, 50],
    "Exact_Match": [0.02, 0.15, 0.30, 0.38, 0.39],
    "Mean_Tanimoto": [0.45, 0.72, 0.85, 0.90, 0.91]
}

df_progress = pd.DataFrame(checkpoint_data)

plt.figure(figsize=(10, 5))
plt.plot(df_progress["Epoch"], df_progress["Exact_Match"], marker='o', color='blue', label='Exact Match %')
plt.title("Exact Match Accuracy vs. Training Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.savefig("exact_match_trend.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df_progress["Epoch"], df_progress["Mean_Tanimoto"], marker='s', color='green', label='Mean Tanimoto')
plt.title("Mean Tanimoto Similarity vs. Training Epoch")
plt.xlabel("Epoch")
plt.ylabel("Tanimoto Score")
plt.grid(True, alpha=0.3)
plt.savefig("tanimoto_trend.png")
plt.show()




