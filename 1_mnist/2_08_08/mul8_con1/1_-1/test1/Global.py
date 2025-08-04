import numpy as np
import torch

N_fn=8
N_con=1
N_ent=8
N_map=16 # (N_con*N_ent)*(N_con*N_ent)
N_pix1=int((N_fn-N_con)/2+1)
N_pix2=(N_fn*N_ent)*(N_fn*N_ent)
N_par=N_con*N_ent
N_Ite=1000
learning_rate=1e-1
Batch=32

Dtype=torch.float
Dtypec=torch.cfloat
dev="cpu"


