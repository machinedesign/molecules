"""
this module provides utility functions related to molecules,
including molecule descriptors.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from .sascorer import calculateScore
from .qed import weights_none
from .qed import weights_mean
from .qed import weights_max


def logp(s):
    """
    computes the logP descriptor a SMILES representation of the molecule.
    check <http://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors>
    """
    mol = _mol_from_smiles(s)
    return Descriptors.MolLogP(mol)


def synthetic_accessibility(s):
    """
    computes synthetic acessibility, see:
    <http://www.rdkit.org/docs/Overview.html> "SA_Score: Synthetic assessibility score"
    """
    mol = _mol_from_smiles(s)
    return calculateScore(mol)


def canonical(s):
    """
    takes a SMILES molecule, convert it to a SMILES canonical molecule to avoid
    graph isomorphism probs.
    """
    mol = _mol_from_smiles(s)
    s = Chem.MolToSmiles(mol)
    return s


def is_valid(s):
    """ True if a SMILES molecule is valid syntaxically and semantically """
    mol = _mol_from_smiles(s)
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
    mol = _mol_from_smiles(s)
    if weights == 'none':
        return weights_none(mol)
    elif weights == 'mean':
        return weights_mean(mol)
    elif weights == 'max':
        return weights_max(mol)
    else:
        raise ValueError(
            'Expected weights to be "none", "mean" or "max", got : "{}"'.format(weights))


def ring_penalty(s, ring_length_thresh=6):
    """
    ring penalty used in [1].
    [1]Automatic chemical design using a data-driven continuous
       representation of molecules
    """
    mol = _mol_from_smiles(s)
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    if len(rings):
        max_ring_length = max(map(len, rings))
        if max_ring_length > ring_length_thresh:
            return max_ring_length
        else:
            return 0
    else:
        return 0

def circular_fingerprint(s, radius=2, n_bits=1024):
    """
    Computes circular fingerprint[1] with the given radius and number of bits.
    Fingerprints are boolean vectors representing molecules.
    They can be  used for molecule similarity or as features 
    for building machine learning classifiers or regressors.

    with radius=2 and n_bits=1024 (the default), it implements the fingerprint named 
    ECFP4, which is the one used in [2].

    [1] Rogers, D.; Hahn, M. “Extended-Connectivity Fingerprints.” J. Chem. Inf. and Model. 50:742-54 (2010).
    [2] Generating Focussed Molecule Libraries for Drug Discovery with Recurrent Neural Networks
        Segler, Marwin HS and Kogej, Thierry and Tyrchan, Christian and Waller, Mark P.

    Returns
    -------

    a list of 'n_bits' size, where each element is 0 or 1.
    """
    m = _mol_from_smiles(s)
    vect = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits).ToBitString()
    vect = map(int, vect)
    vect = list(vect)
    return vect

def draw_image(s, filename, size=(300, 300)):
    """takes a SMILES molecule and draw it in an png in filename. size is in pixels"""
    mol = _mol_from_smiles(s)
    Draw.MolToFile(mol, filename, size=size)


def _mol_from_smiles(s):
    """encapsulated here to allow for instance caching"""
    return Chem.MolFromSmiles(s)
