import pickle

import pandas as pd


class PreprocessData:
    def __init__(self, file_path):
        with open(file_path, 'rb') as handle:
            self.data = pickle.load(handle)

    def process(self, report=False):
        l_dict = len(self.data[0])
        if l_dict == 2:
            df = pd.DataFrame.from_dict(self.data, orient='index').rename(columns={0: 'idx', 1: 'list'})
            df[['L', 'H', 'm', 'C', 'cut', 'a', 'b']] = df.list.to_list()
            df.drop(columns=['list'], inplace=True)
            if report is True:
                print("Unique bin types:", len(df['idx'].unique()))
            return df
        elif l_dict == 6:
            df = pd.DataFrame.from_dict(self.data, orient='index')
            df = df.rename(columns={0: 'l', 1: 'h', 2: 'r+', 3: 'f', 4: 'rho', 5: 'phi'})
            if report is True:
                print("Unique items:", len(df))
            return df
        else:
            raise ValueError("Unrecognized data format imported. Please pass a pickled file with data for set of bins "
                             "or set of items")


if __name__ == "__main__":
    B = "B.pickle"
    R = "R.pickle"

    B_data = PreprocessData(B).process(report=True)
    R_data = PreprocessData(R).process(report=True)
