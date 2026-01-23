from scipy.io import loadmat
import pprint

mat_path = r"C:\Users\abhin\OneDrive\Documents\GitHub\SemiETS-Improvements\data\totaltext\annotations"

import os
files = [f for f in os.listdir(mat_path) if f.endswith(".mat")]
print("Using file:", files[0])

mat = loadmat(os.path.join(mat_path, files[0]))

print("\n=== TOP-LEVEL KEYS ===")
print(mat.keys())

print("\n=== FULL STRUCTURE (truncated) ===")
pprint.pprint(mat)
