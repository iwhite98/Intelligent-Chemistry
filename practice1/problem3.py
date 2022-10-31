from rdkit import Chem
import numpy as np

f = open("filter_1.txt", "r")
mol_arr = np.empty(0, str)
atom_arr = np.zeros((0,100))
atom_sym = np.empty(100,str)
mol_i = 0

while True:
    smiles = f.readline()
    if not smiles:
        break
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    mol_arr = np.append(mol_arr, mol)
    atom_arr = np.append(atom_arr, np.zeros((1,100)), axis = 0)
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        atom_num = atom.GetAtomicNum()
        atom_arr[mol_i][atom_num]+=1
    mol_i += 1
    sum = atom_arr.sum()
    num_H = atom_arr.sum(axis=0)[1]

for i in range(mol_i):
    print("distribution of", Chem.MolToSmiles(mol_arr[i]))
    for j in range(100):
        if(atom_arr[i][j] != 0):
            atom = Chem.Atom(j)
            print(atom.GetSymbol(), " without Hydrogen : ",int(atom_arr[i][j]),"/",int(sum-num_H), " add Hydrogen : ", int(atom_arr[i][j]),"/",int(sum) )

f.close()
