import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from hgraph import HierVAE, common_atom_vocab, PairVocab, MolGraph

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'ckpt/antibiotc/model.ckpt.300'
VOCAB_PATH = 'data/scraped_data/vocab.txt'
DATA_PATH = 'data/scraped_data/scrapedData_clean.txt'

def load_model_and_vocab():
    print(f"📦 Loading Vocabulary from {VOCAB_PATH}...")
    with open(VOCAB_PATH) as f:
        vocab_lines = [line.strip().split() for line in f if line.strip()]
        vocab = PairVocab(vocab_lines, cuda=(DEVICE == 'cuda'))

    args = argparse.Namespace(
        vocab=vocab, atom_vocab=common_atom_vocab, rnn_type='LSTM',
        hidden_size=250, embed_size=250, latent_size=32,
        depthT=15, depthG=15, diterT=1, diterG=3, dropout=0.0
    )

    model = HierVAE(args).to(DEVICE)
    print(f"⬇️  Loading Checkpoint from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, vocab, args

def extract_properties(mol):
    return {
        "MW": Descriptors.MolWt(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol)
    }

def visualize_probing_results(df):
    """Generates a professional bar plot of the R2 scores."""
    df_plot = df.melt(id_vars="Stage", var_name="Chemical Property", value_name="R2 Score")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    palette = sns.color_palette("viridis", n_colors=2)
    ax = sns.barplot(data=df_plot, x="Stage", y="R2 Score", hue="Chemical Property", palette=palette)
    
    plt.title("Hierarchical Probing: $R^2$ Scores across Encoder Layers", fontsize=15, pad=20)
    plt.ylim(0, 1.1)
    plt.ylabel("Predictive Accuracy ($R^2$)", fontsize=12)
    plt.xlabel("Model Hierarchy Stage", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig("probing_results.png", dpi=300)
    print("📈 Plot saved as 'probing_results.png'")
    plt.show()

def run_probing_experiment():
    model, vocab, args = load_model_and_vocab()
    
    with open(DATA_PATH) as f:
        smiles_list = [l.strip() for l in f.readlines()[:500]]

    probe_features = {"Latent_Z": [], "Tree_Hidden": [], "Graph_Hidden": []}
    probe_targets = {"MW": [], "Rings": []}

    print(f"🧪 Extracting Hidden States on {DEVICE}...")

    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            
            _, all_tensors, _ = MolGraph.tensorize([smi], vocab, common_atom_vocab)
            tree_tensors = [t.to(DEVICE) if torch.is_tensor(t) else t for t in all_tensors[0]]
            graph_tensors = [t.to(DEVICE) if torch.is_tensor(t) else t for t in all_tensors[1]]
            
            with torch.no_grad():
                hroot, hnode, hinter, hatom = model.encoder(tree_tensors, graph_tensors)
                
                probe_features["Latent_Z"].append(hroot.cpu().numpy().flatten())
                probe_features["Tree_Hidden"].append(hnode.mean(dim=0).cpu().numpy().flatten())
                probe_features["Graph_Hidden"].append(hatom.mean(dim=0).cpu().numpy().flatten())

                props = extract_properties(mol)
                probe_targets["MW"].append(props["MW"])
                probe_targets["Rings"].append(props["Rings"])
        except Exception:
            continue

    final_results = []
    print(f"📊 Training Ridge Probes...")

    for stage, X in probe_features.items():
        X = np.array(X)
        stage_metrics = {"Stage": stage}
        
        for prop_name, y_values in probe_targets.items():
            y = np.array(y_values)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            clf = Ridge(alpha=1.0)
            clf.fit(X_train, y_train)
            score = r2_score(y_test, clf.predict(X_test))
            stage_metrics[prop_name] = max(0, round(score, 4)) # Ensure no negative R2 in display
            
        final_results.append(stage_metrics)

    df = pd.DataFrame(final_results)
    df.to_csv("probing_metrics.csv", index=False)
    
    print("\n" + "="*50)
    print("      HIERARCHICAL PROBING RESULTS (R²)")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)

    visualize_probing_results(df)

if __name__ == "__main__":
    run_probing_experiment()