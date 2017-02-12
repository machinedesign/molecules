import numpy as np

from clize import run

def remove_intersection(filename_dest, filename_source, out):
    X = np.load(filename_dest)['X']
    X = set(X)
    Y = np.load(filename_source)['X']
    Y = set(Y)
    X = X - Y
    print(X & Y)
    X = list(X)
    print(len(X))
    np.savez(out, X=X)

if __name__ == '__main__':
    run(remove_intersection)
