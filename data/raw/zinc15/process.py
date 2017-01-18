if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('zinc15.txt', delimiter = '\t')
    df = df[['smiles']]
    print(df.columns)
    df.to_csv('zinc15.csv')
