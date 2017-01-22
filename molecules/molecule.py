"""
this module provides utility functions related to molecules,
including molecule descriptors.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from .sascorer import calculateScore
from .qed import weights_none
from .qed import weights_mean
from .qed import weights_max

def logp(s):
    """
    computes the logP descriptor a SMILES representation of the molecule.
    check <http://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors>
    """
    mol = Chem.MolFromSmiles(s)
    return Descriptors.MolLogP(mol)


def synthetic_accessibility(s):
    """
    computes synthetic acessibility, see:
    <http://www.rdkit.org/docs/Overview.html> "SA_Score: Synthetic assessibility score"
    """
    mol = Chem.MolFromSmiles(s)
    return calculateScore(mol)


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

def qed(s, weights='none'):
    """ 
    Return the quantitative estimation of drug-likeness (QED) of a molecule from [1].
    QED consist of a weighted average of several molecule properties, 'weights' define
    how we combine the properties:
        - if weights is 'none' : the properties have all weight 1
        - if weights is 'mean': use the 'average' descriptor weights defined in [1]
        - if weights is 'max': use the 'maximal' descriptor weights defined in [1]
    [1] Uoyama, H., Goushi, K., Shizu, K., Nomura, H. & Adachi, C. Highly efficient organic
    light-emitting diodes from delayed fluorescence. Nature 492, 234–238 (2012). URL
    http://dx.doi.org/10.1038/nature11687.
    """
    mol = Chem.MolFromSmiles(s)
    if weights == 'none':
        return weights_none(mol)
    elif weights == 'mean':
        return weights_mean(mol)
    elif weights == 'max':
        return weights_max(mol)
    else:
        raise ValueError('Expected weights to be "none", "mean" or "max", got : "{}"'.format(weights))

def draw_image(s, filename, size=(300, 300)):
    """takes a SMILES molecule and draw it in an png in filename. size is in pixels"""
    mol = Chem.MolFromSmiles(s)
    Draw.MolToFile(mol, filename, size=size)
