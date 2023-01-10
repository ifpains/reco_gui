import numpy as np
from glob import glob
from root_pandas import read_root
import pandas as pd


def hypot(x, y):
    ax = float(np.abs(x))
    ay = float(np.abs(y))
    amax = 0
    amin = 0
    if ax > ay:
        amax = ax
        amin = ay
    else:
        amin = ax
        amax = ay

    if amin == 0:

        return amax

    f = amin/amax
    return amax*np.sqrt(1.0 + f*f)


def Hypot(x, y):
    return hypot(x, y) + 0.5

class RootReader:
    def __init__(self, path:str, columns:list) -> None:
        self.path = path
        self.columns = columns
    
    def get_data(self) -> pd.DataFrame:
        df_file = read_root(self.path, columns=self.columns)
        df_file = df_file.drop_duplicates(['run', 'event'], 
                                          keep='first')
        return df_file

    def transform(self) -> pd.DataFrame:
        ## TODO do it usind only data
        df = self.get_data()
        df = df[df['nCl'] > 0]
        columns_of_lists = ['cl_integral', 'cl_length', 
                            'cl_width', 'cl_nhits', 
                            'cl_iteration', 'cl_xmean', 
                            'cl_ymean', 'cl_size']
        simple_columns = [i for i in df.columns if i not in columns_of_lists]
        flatten_data = []
        # explode list colums
        for c in columns_of_lists:
            flatten_data.append(
                list(np.concatenate(np.array(df[c].values.tolist()))))
        struct_data_values = np.array(flatten_data).T

        struct_data_dataframe = pd.DataFrame(
            struct_data_values, columns=columns_of_lists)

        const_data = df[simple_columns]
        const_data = const_data.loc[const_data.index.repeat(
        const_data.nCl)].reset_index(drop=True)
        transformed_data = pd.concat([const_data, struct_data_dataframe], axis=1)
        transformed_data['roi'] = transformed_data.apply(
        lambda x: Hypot(x.cl_xmean - 1024, (x.cl_ymean - 1024)*1.2), axis=1)
        transformed_data['slimness'] = transformed_data['cl_width'] / \
            transformed_data['cl_length']

        transformed_data['filter_name'] = self.path.split(
            '/')[-1].split('.')[0].split('_')[-4]
        transformed_data['n_pts'] = self.path.split(
            '/')[-1].split('.')[0].split('_')[-3]
        
        return transformed_data
