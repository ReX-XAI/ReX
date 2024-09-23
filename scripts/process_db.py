#!/usr/bin/env python
from PIL import Image
import pandas as pd
import numpy as np
import zlib
import os
import sys

def dataframe(db, table):
    return pd.read_sql_table(table, f"sqlite:///{db}")


def to_numpy(buffer, dtype, shape):
    return np.frombuffer(zlib.decompress(buffer), dtype=dtype).reshape(shape)

# def outside_segmentation(mask, segment) -> int:
#     mask = mask.detach().cpu().numpy().astype(np.uint8)
#     segment = segment.detach().cpu().numpy().astype(np.uint8)
#     segment = np.where(segment > 0, 1, 0)
#     return (np.count_nonzero(mask) - np.count_nonzero(mask + segment == 2)) // 3

def find(name, path):
    for root, dirs, files in os.walk(path):
        sn = os.path.splitext(os.path.basename(name))[0]
        for f in files:
            if sn == os.path.splitext(os.path.basename(f))[0]:
                return os.path.join(root, f)

def do(df, segs):
    output = []

    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        name = os.path.basename(row['path'])
        exp = to_numpy(row['explanation'], np.float32, (224, 224, 3))
        exp = np.where(exp > 0, 1, 0)
        seg = find(name, segs)
        if seg is not None:
            segment = np.array(Image.open(seg).convert("RGB").resize((224, 224)))
            segment = np.where(segment > 0, 1, 0)
            outside = (np.count_nonzero(exp) - np.count_nonzero(exp + segment == 2)) // 3
            output.append((name, outside))
    return output

def do_photobomb(df, segs):
    output = []

    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        name = os.path.basename(row['path'])
        print(name)
        exp = to_numpy(row['explanation'], np.float32, (224, 224, 3))
        exp = np.where(exp > 0, 1, 0)
        seg = find(name, segs)
        print(seg)
        if seg is not None:
            segment = np.array(Image.open(seg).convert("RGB").resize((224, 224)))
            segment = np.where(segment > 0, 1, 0)
            inside = np.count_nonzero(exp + segment == 2) // 3
            output.append((name, inside))
    return output


if __name__ == "__main__":
    db = sys.argv[1]
    segs = sys.argv[2]


    if db.endswith(".db"):
        df = dataframe(db, 'rex')
        output = do_photobomb(df, segs)
        # output = do(df, segs)
        with open('rex_out.csv', "a") as f:
            for n, o in output:
                f.write(f"{n},{o}\n")
