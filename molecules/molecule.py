"""
this module provides utility functions related to molecules,
including molecule descriptions.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw


def logp(s):
    """
    computes the logP descriptor a SMILES representation of the molecule.
    check <http://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors>
    """
    mol = Chem.MolFromSmiles(s)
    return Descriptors.MolLogP(mol)


def canonical(s):
    """
    takes a SMILES molecule, convert it to a SMILES canonical molecule to avoid
    graph isomorphism probs.
    """
    mol = Chem.MolFromSmiles(s)
    s = Chem.MolToSmiles(mol)
    return s


def is_valid(s):
    """ True if a SMILES molecule is valid syntaxically and semantically """
    mol = Chem.MolFromSmiles(s)
    return mol is not None


def draw_image(s, filename, size=(300, 300)):
    """takes a SMILES molecule and draw it in an png in filename. size is in pixels"""
    mol = Chem.MolFromSmiles(s)
    Draw.MolToFile(mol, filename, size=size)
