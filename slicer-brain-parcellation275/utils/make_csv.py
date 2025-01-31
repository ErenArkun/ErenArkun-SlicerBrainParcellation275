import numpy as np
import pandas as pd
import os

def make_csv(parcellation, save):
    script_dir = os.path.dirname(os.path.realpath(__file__))

    csv_path = os.path.join(script_dir, "..", "level" , "Level5.txt")
    df = (
        pd.read_table(csv_path, names=["number", "region"])
        .astype("str")
        .set_index("number")
    )
    for i in range(280):
        i += 1
        volume = np.count_nonzero(parcellation == i)
        df.loc[str(i), save] = volume
    df = df.set_index("region").T.reset_index().rename(columns={"index": "uid"})
    return df
