from rdkit import Chem
from rdkit.Chem import Descriptors

def logp(s):
    mol = Chem.MolFromSmiles(s)
    return Descriptors.MolLogP(mol)

def canonical(s):
    mol = Chem.MolFromSmiles(s)
    s = Chem.MolFromSmiles(mol)
    return s
