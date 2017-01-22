if __name__ == '__main__':
    import pandas as pd
    df = pd.read_table('chembl_22_chemreps.txt.gz',compression='gzip')
    df = df.rename(columns={'canonical_smiles':'smiles'})
    df.to_csv('chembl22.csv')
