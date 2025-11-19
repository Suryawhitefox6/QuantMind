import pickle
import numpy as np
import matplotlib.pyplot as plt

HEBB_PATH = "hebbian.pkl"

def hebbian_dynamics():
    hebb = pickle.load(open(HEBB_PATH, "rb"))
    cs = hebb.get("chunk_strength", {})
    if not cs:
        print("No Hebbian data found yet.")
        return

    values = list(cs.values())
    print(f"🧩 Total updated chunks: {len(values)}")
    print(f"Mean Hebbian Strength: {np.mean(values):.4f}")
    print(f"Max Hebbian Strength: {np.max(values):.4f}")

    plt.hist(values, bins=20)
    plt.title("Distribution of Hebbian Chunk Strengths")
    plt.xlabel("Strength")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    hebbian_dynamics()
