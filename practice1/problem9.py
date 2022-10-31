from rdkit import Chem

file = open("filter_1.txt", "r")

while True:
    smiles = file.readline()
    if not smiles:
        break
    mol = Chem.MolFromSmiles(smiles)
    print("Make CNN of : ", smiles,end=' ')
    for i in range(mol.GetNumAtoms()):
        atom_num = mol.GetAtomWithIdx(i).GetAtomicNum()
        print(atom_num-6,end='')
    print("\n")

file.close()
