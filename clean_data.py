import os
import sys
from rdkit import Chem
from tqdm import tqdm

# Ensure the hgraph module is discoverable
sys.path.append(os.getcwd())
try:
    from hgraph import MolGraph
except ImportError:
    print("Error: Could not import MolGraph. Ensure you are in the hgraph2graph root directory.")
    sys.exit(1)

# Constants
INPUT_FILE = 'data/scraped_data/scrapedData.txt'
OUTPUT_FILE = 'data/scraped_data/scrapedData_clean.txt'

def validate_molecule(smi):
    """
    Runs a multi-stage validation to ensure the molecule is compatible
    with the Hierarchical Graph logic.
    """
    try:
        # Stage 1: RDKit Parse
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        
        # Stage 2: Kekulization (Critical for hgraph2graph aromaticity handling)
        # We use a copy to avoid modifying the original during test
        mol_copy = Chem.Mol(mol)
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        
        # Stage 3: HGraph Junction Tree Decomposition
        # This is the "Stress Test" - it checks if the molecule can be 
        # broken into valid clusters/fragments.
        _ = MolGraph(smi)
        
        # Stage 4: Canonicalization
        return Chem.MolToSmiles(mol, isomericSmiles=False)
        
    except Exception:
        return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return

    # Load unique SMILES to avoid redundant processing
    with open(INPUT_FILE, 'r') as f:
        raw_smiles = [line.strip().split()[0] for line in f if line.strip()]
    
    unique_smiles = list(dict.fromkeys(raw_smiles))
    print(f"Validating {len(unique_smiles)} unique molecules against HGraph logic...")

    clean_smiles = []
    
    # Using tqdm for a progress bar (standard in ML workflows)
    for smi in tqdm(unique_smiles, desc="Cleaning"):
        valid_smi = validate_molecule(smi)
        if valid_smi:
            clean_smiles.append(valid_smi)

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(clean_smiles) + '\n')

    print(f"\nValidation Complete:")
    print(f"- Original: {len(raw_smiles)}")
    print(f"- Passed:   {len(clean_smiles)}")
    print(f"- Rejected: {len(raw_smiles) - len(clean_smiles)}")
    print(f"Clean data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()