import numpy as np


def load_data_from_df(df, col_path, col_label):
    import nibabel as nib
    data = np.zeros((df.shape[0], 121, 145, 121, 1))
    for j, (_, el) in enumerate(df.iterrows()):
        data[j, :, :, :, 0] = nib.load(el[col_path]).get_fdata()
    return data, df[col_label].to_numpy()

