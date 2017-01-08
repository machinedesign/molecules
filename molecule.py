"""
this module provides utility functions related to molecules,
including molecule descriptions.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw

def logp(s):
    mol = Chem.MolFromSmiles(s)
    return Descriptors.MolLogP(mol)

def canonical(s):
    mol = Chem.MolFromSmiles(s)
    s = Chem.MolToSmiles(mol)
    return s

def is_valid(s):
    mol = Chem.MolFromSmiles(s)
    return mol is not None

def draw_image(s, filename):
    mol = Chem.MolFromSmiles(s)
    Draw.MolToFile(mol, filename)
