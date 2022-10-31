from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np

file = open("smiles.txt", "r")

while True:
    smiles = file.readline()
    if not smiles:
        break
    
    atom_arr = np.empty(0, str)
    atom_feature = np.zeros((0,3))
    
    mol = Chem.MolFromSmiles(smiles)
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if(np.all(atom_arr != atom.GetSymbol())):
            index = len(atom_arr)
            atom_arr = np.append(atom_arr, atom.GetSymbol())
            atom_feature = np.insert(atom_feature, index , 0 , axis=1)
        else:
            index = np.where(np.isin(atom_arr, atom.GetSymbol()) == True)[0][0]
    
        atom_feature = np.append(atom_feature, np.zeros((1, len(atom_arr)+3)), axis=0)
        atom_feature[i][index] = 1
        
        if(atom.GetHybridization() == Chem.HybridizationType.SP):
            atom_feature[i][len(atom_arr)] = 1
        elif(atom.GetHybridization() == Chem.HybridizationType.SP2):
            atom_feature[i][len(atom_arr)+1] = 1
        elif(atom.GetHybridization() == Chem.HybridizationType.SP3):
            atom_feature[i][len(atom_arr)+2] = 1
        
        GNN = (atom_feature, rdmolops.GetAdjacencyMatrix(mol) )

        print("GNN of ", smiles, " : " ,GNN)
