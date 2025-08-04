import numpy as np
import torch

Dtype=torch.float
Dtypec=torch.cfloat

N_fn=8
qubit=12
N_con=2
N_ent=8
N_map=16
N_pix1=int((N_fn-N_con)/2+1)
N_pix2=(N_fn*N_ent)*(N_fn*N_ent)
N_par=N_con*N_ent


