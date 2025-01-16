import os
import numpy as np

fout = "tmp.out"

for buffering in [-1, 0]:
    for i in range(10,21):
        D = np.zeros(3*2**i, dtype=np.double)
        print(f'Writing array of size {D.nbytes} to {fout} with buffering set to {buffering}', flush=True)
        with open(fout, "wb", buffering=buffering) as file:
            for j in range(10):
                file.write(D)
            

try:
    os.remove(fout)
except FileNotFoundError as e:
    pass
