import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*') 
input_path = 'data/scraped_data/scrapedData_clean.txt'
output_csv = 'results.csv'
np.random.seed(42)

def model_predict(smiles_in):
    """
    Simulates model performance based on your reported ~80% validity.
    REPLACE THIS with: return your_model.reconstruct(smiles_in)
    """
    mol = Chem.MolFromSmiles(smiles_in)
    if not mol: return ""
    chance = np.random.rand()
    if chance > 0.80: return "INVALID" 
    if chance > 0.40: return smiles_in  
    s = Chem.MolToSmiles(mol)
    return s.replace("C", "N", 1) if "C" in s else s

if not os.path.exists(input_path):
    raise FileNotFoundError(f"File not found: {input_path}")

with open(input_path, 'r') as f:
    smiles_list = [line.strip() for line in f if line.strip()]

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

results_data = []
print(f"Starting evaluation on {len(smiles_list)} molecules...")

for i, sm_in in enumerate(smiles_list):
    sm_out = model_predict(sm_in)
    
    m_in = Chem.MolFromSmiles(sm_in)
    m_out = Chem.MolFromSmiles(sm_out) if sm_out != "INVALID" else None
    
    res = {
        "id": i,
        "smiles_in": sm_in,
        "smiles_out": sm_out if m_out else "",
        "valid": 1 if m_out else 0,
        "exact_match": 0,
        "tanimoto": np.nan,
        "delta_atoms": np.nan,
        "delta_bonds": np.nan
    }
    
    if m_out:
        can_in = Chem.MolToSmiles(m_in, isomericSmiles=True)
        can_out = Chem.MolToSmiles(m_out, isomericSmiles=True)
        res["exact_match"] = 1 if can_in == can_out else 0
        
        fp_in = fp_gen.GetFingerprint(m_in)
        fp_out = fp_gen.GetFingerprint(m_out)
        res["tanimoto"] = DataStructs.TanimotoSimilarity(fp_in, fp_out)
        
        res["delta_atoms"] = m_out.GetNumAtoms() - m_in.GetNumAtoms()
        res["delta_bonds"] = m_out.GetNumBonds() - m_in.GetNumBonds()
        
    results_data.append(res)

df = pd.DataFrame(results_data)
df.to_csv(output_csv, index=False)

print("\n--- PERFORMANCE SUMMARY ---")
summary = pd.DataFrame({
    "Metric": ["Validity Rate", "Exact Accuracy", "Mean Tanimoto", "Avg Atom Delta"],
    "Value": [
        f"{df['valid'].mean():.2%}",
        f"{df['exact_match'].mean():.2%}",
        f"{df['tanimoto'].mean():.3f}",
        f"{df['delta_atoms'].abs().mean():.2f}"
    ]
})
print(summary.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.histplot(df['tanimoto'].dropna(), bins=20, kde=True, color='teal')
plt.title('Similarity Distribution (Tanimoto Scores)')
plt.xlabel('Tanimoto Similarity')
plt.ylabel('Frequency')
plt.savefig('similarity_distribution.png')

print("\nGenerating qualitative grid...")
subset = df.head(10)
draw_mols, draw_legends = [], []

for _, row in subset.iterrows():
    draw_mols.append(Chem.MolFromSmiles(row['smiles_in']))
    if row['valid']:
        draw_mols.append(Chem.MolFromSmiles(row['smiles_out']))
        draw_legends.extend([f"In {row['id']}", f"Out {row['id']} (T:{row['tanimoto']:.2f})"])
    else:
        draw_mols.append(Chem.MolFromSmiles("")) 
        draw_legends.extend([f"In {row['id']}", f"ID {row['id']} INVALID"])

img = Draw.MolsToGridImage(draw_mols, molsPerRow=2, subImgSize=(300, 300), legends=draw_legends)
img.save('qualitative_grid.png')
print("Done. Check 'results.csv', 'similarity_distribution.png', and 'qualitative_grid.png'.")