import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

def loadSmilesAndSave(smis, path):
    """Render one SMILES string to a 224 x 224 molecular image."""
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224,224))
    img.save(path)

def main():
    """Generate the drug images and their filename mapping."""
    parser = argparse.ArgumentParser(description='Generate EPRD-DDI molecular images')
    parser.add_argument('--dataroot', type=str, default="./datasets/", help='data root')
    args = parser.parse_args()

    raw_file_path = os.path.join(args.dataroot, "drugbank", "drug_smiles.csv")
    img_save_root = os.path.join(args.dataroot, "drug_image", "drugbank")
    csv_save_path = os.path.join(args.dataroot,  "drug_image.csv")
    error_save_path = os.path.join(args.dataroot, "drugbank", "error_smiles.csv")
    if not os.path.exists(img_save_root):
        os.makedirs(img_save_root)

    df = pd.read_csv(raw_file_path)
    drug_id, smiles = df["drug_id"].values, df["smiles"].values

    processed_ac_data = []
    error_smiles = []
    for i, s in tqdm(zip(drug_id, smiles), total=len(drug_id)):
        filename = "{}.png".format(i)
        img_save_path = os.path.join(img_save_root, filename)
        try:
            loadSmilesAndSave(s, img_save_path)
            processed_ac_data.append([i, filename])
        except Exception:
            print(f'error smiles : {s}')
            error_smiles.append(s)

    processed_ac_data = np.array(processed_ac_data)
    pd.DataFrame({
        "drug_id": processed_ac_data[:, 0],
        "filename": processed_ac_data[:, 1],
    }).to_csv(csv_save_path, index=False)

    if len(error_smiles) > 0:
        pd.DataFrame({"smiles": error_smiles}).to_csv(error_save_path, index=False)

if __name__ == '__main__':
    main()
