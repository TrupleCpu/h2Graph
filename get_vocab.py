import sys
import argparse
from hgraph import MolGraph
from rdkit import Chem
from multiprocessing import Pool

def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        try:
            # Creating the MolGraph object
            hmol = MolGraph(s)

            # Adding labels to the vocabulary
            for node, attr in hmol.mol_tree.nodes(data=True):
                smiles = attr['smiles']
                vocab.add(attr['label'])
                for i, s in attr['inter_label']:
                    vocab.add((smiles, s))

        except Exception as e:
            # Handle any errors during processing (e.g., invalid SMILES)
            print(f"Error processing {s}: {e}")
            continue

    return vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1, help='Number of CPU cores to use')
    args = parser.parse_args()

    # Read the data from stdin and prepare the list
    data = [line.strip() for line in sys.stdin if line.strip()]
    data = list(set(data))  # Remove duplicate lines

    # Split the data into batches for parallel processing
    batch_size = len(data) // args.ncpu + (len(data) % args.ncpu != 0)  # Round up for batch size
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    # Create a pool of worker processes and process the data
    with Pool(args.ncpu) as pool:
        vocab_list = pool.map(process, batches)

    # Flatten the list of vocab sets and remove duplicates
    vocab = set([item for vocab in vocab_list for item in vocab])

    # Sort and print the vocabulary
    for x, y in sorted(vocab):
        print(x, y)

if __name__ == "__main__":
    main()
