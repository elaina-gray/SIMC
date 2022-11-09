import os.path as op
import numpy as np
import pandas as pd


class InputDataPreparation:
    def __init__(self):
        self.df = None

    def normalise_raw_data(self):
        raw_year_1 = pd.read_csv('/Users/wennnn/PycharmProjects/SIMC/data/raw/data_year_1.csv')
        raw_year_2 = pd.read_csv('/Users/wennnn/PycharmProjects/SIMC/data/raw/data_year_2.csv')
        raw_year_3 = pd.read_csv('/Users/wennnn/PycharmProjects/SIMC/data/raw/data_year_3.csv')
        raw_all = pd.concat([raw_year_1, raw_year_2, raw_year_3])
        raw_all.to_csv('/Users/wennnn/PycharmProjects/SIMC/data/raw/raw_all.csv')
        raw_all = raw_all.drop(columns=['Dates'])
        raw_all.astype('float')
        raw_all = raw_all.div(500)
        raw_all = raw_all.reset_index()
        raw_all = raw_all.drop(columns=['index'])
        self.df = raw_all
        raw_all.to_csv('/Users/wennnn/PycharmProjects/SIMC/data/raw/normalised.csv')

    def reshape_data(self, starting):
        self.df = pd.read_csv('/Users/wennnn/PycharmProjects/SIMC/data/raw/normalised.csv')
        locs = pd.read_csv('/Users/wennnn/PycharmProjects/SIMC/data/raw/locations.csv')
        final = np.empty([761-starting-20, 20, 26, 5])
        for day in range(0, 761-starting-20):
            for history in range(0, 20):
                for location in range(0, 26):
                    final[day][history][location][0] = locs.iloc[0, location]
                    final[day][history][location][1] = locs.iloc[1, location]
                    final[day][history][location][2] = self.df.iloc[day+history][location+1]
                    final[day][history][location][3] = self.df.iloc[day+history][location+26+1]
                    final[day][history][location][4] = self.df.iloc[day+history][location+52+1]
        np.save('/Users/wennnn/PycharmProjects/SIMC/data/processed/4d_array.npy', final)
        print(final)

    def output_data(self):
        self.df = pd.read_csv('/Users/wennnn/PycharmProjects/SIMC/data/raw/normalised.csv')
        locs = pd.read_csv('/Users/wennnn/PycharmProjects/SIMC/data/raw/locations.csv')
        final = np.empty([761-20, 26, 3])
        for day in range(0, 761-20):
            for location in range(0, 26):
                final[day][location][0] = self.df.iloc[day+20][location+1]
                final[day][location][1] = self.df.iloc[day+20][location+26+1]
                final[day][location][2] = self.df.iloc[day+20][location+52+1]
        np.save('/Users/wennnn/PycharmProjects/SIMC/data/processed/target.npy', final)

    def print_array(self):
        print(np.load('/Users/wennnn/PycharmProjects/SIMC/data/processed/4d_array.npy'))


if __name__ == '__main__':
    class_object = InputDataPreparation()
    class_object.output_data()


