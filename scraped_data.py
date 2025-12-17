import requests
import pandas as pd
import time
import os


# Create data directory if it doesn't exist
def create_data_directory():
    os.makedirs('data/scraped_data', exist_ok=True)


# Function to check for patents using a SMILES string
def check_patent_by_smiles(smiles, base_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug", headers=None, timeout=10):
    """Checks PubChem for patents using a SMILES string."""
    if headers is None:
        headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        # Step 1: Find CID by SMILES
        res = requests.post(f"{base_url}/compound/smiles/cids/JSON", 
                            data={'smiles': smiles}, 
                            headers=headers, timeout=timeout).json()
        cid = res['IdentifierList']['CID'][0]  # Extract first CID

        # Step 2: Check for Patents
        patent_req = requests.get(f"{base_url}/compound/cid/{cid}/xrefs/Patent/JSON", 
                                  headers=headers, timeout=timeout).json()
        
        return 1 if 'InformationList' in patent_req else 0
    except Exception as e:
        print(f"Error checking patent for SMILES {smiles}: {e}")
        return 0


# Function to load SMILES data from file
def load_smiles_data(file_path, start_index=300, end_index=500):
    """Loads SMILES data from a file and returns a list of SMILES."""
    try:
        with open(file_path, 'r') as f:
            # Read lines between start_index and end_index (inclusive)
            return [line.strip() for i, line in enumerate(f.readlines()) if start_index <= i < end_index]
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Check your folder structure!")
        exit()


# Function to check patents for a list of SMILES
def check_patents_for_smiles(smiles_list, delay=0.4):
    """Checks patents for each molecule in the list of SMILES."""
    results = []
    print(f"Checking patents for {len(smiles_list)} molecules. This will take some time...")

    for i, smiles in enumerate(smiles_list):
        is_patented = check_patent_by_smiles(smiles)
        results.append({"id": i, "smiles": smiles, "is_patented": is_patented})
        
        if i % 10 == 0:
            print(f"Progress: {i}/{len(smiles_list)} molecules checked...")
        
        time.sleep(delay)  # Crucial: prevents PubChem from blocking your IP

    return results


# Function to save results to CSV files
def save_results(results):
    """Saves results to CSV files for further analysis."""
    df = pd.DataFrame(results)
    df.to_csv('data/scraped_data/metadata.csv', index=False)
    df['smiles'].to_csv('data/scraped_data/scrapedData.txt', index=False, header=False)
    print(f"\nSUCCESS! Files created:")
    print(f"- data/scraped_data/scrapedData.txt (For training)")
    print(f"- data/scraped_data/metadata.csv (For evaluation diagnostics)")


# Main function to run the workflow
def main():
    create_data_directory()
    
    # Load SMILES data (300 to 500 range)
    smiles_file_path = 'data/chembl/all.txt'
    print("Reading local ChEMBL data...")
    all_smiles = load_smiles_data(smiles_file_path, start_index=300, end_index=500)
    
    # Check patents for each SMILES string
    results = check_patents_for_smiles(all_smiles)
    
    # Save results to disk
    save_results(results)


if __name__ == "__main__":
    main()
