from six.moves import filter
from six.moves import map

import numpy as np
import pandas as pd

from molecules import molecule

from clize import run


def preprocess(filename, out, *, max_length=120):
    max_length = int(max_length)
    data = pd.read_csv(filename)
    data = data['smiles']
    print(data.shape)
    data = data.values
    data = data.tolist()
    data = filter(lambda s: len(s) <= max_length, data)
    data = filter(molecule.is_valid, data)
    data = map(molecule.canonical, data)
    data = list(data)
    data = np.array(data)
    print(data.dtype)
    np.savez(out, X=data)


if __name__ == '__main__':
    run(preprocess)
