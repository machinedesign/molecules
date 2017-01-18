if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('13_prop.xls', delimiter = '\t')
    df.to_csv('zinc12.csv')
