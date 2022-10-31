from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np

f = open("smiles.txt", "r")
while True:
   smiles = f.readline()
   if not smiles:
    break
   mol = Chem.MolFromSmiles(smiles)
   fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,2,1024)
   arr = np.asarray(fp)
   print(arr)
   exit()
   avg = arr.sum()/1024
   print(avg)
  

f.close()
