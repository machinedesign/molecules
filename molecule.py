"""
this module provides utility functions related to molecules,
including molecule descriptions.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors

def logp(s):
    mol = Chem.MolFromSmiles(s)
    return Descriptors.MolLogP(mol)

def canonical(s):
    mol = Chem.MolFromSmiles(s)
    s = Chem.MolFromSmiles(mol)
    return s

def is_valid(s):
    mol = Chem.MolFromSmiles(s)
    return mol is not None

def draw_image(s, filename):
    mol = Chem.MolFromSmiles(s)
    Chem.Draw.MolToFile(mol, filename)
