import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DiffusionModel import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl

from tqdm import tqdm
import os
from copy import deepcopy
import time

from scipy.spatial import Delaunay

from fit_utils import *

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
#diffusion_model_allreg.pt
model = DiffusionModel().load('models/compare_explicit_low_lr.pt',compiled=True)
sum(p.numel() for p in model.parameters() if p.requires_grad)


hist = fit_model(model,
                 alpha_conn=.1,
                 alpha_diss=.1,
                 alpha_energy= 10,
                 alpha_heat= .1,
                 alpha_max_min= 10,
                 save_best=True,
                 lr=1e-6,
                 betas=(.5,.99),
                 save_path='models/ex_PiGRAND_all_reg.pt')
start_time = time.perf_counter()
correlations = develop_layers(model, obj='pyramid_7_m', plot_dir='./plots/explicit_transfer_pyramid6M')
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Die Ausführung hat {elapsed_time:.4f} Sekunden gedauert.")
