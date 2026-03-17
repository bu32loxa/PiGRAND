import numpy as np
import torch
import pickle as pkl
from tqdm import tqdm
import os

"""
retrieve graph from simplicial complex and identify vertex classes
"""

obj = 'pyramid_8'
min_layer, max_layer = 571, 1079

#obj = 'Benchmark'
#min_layer, max_layer = 0, 10

files = os.listdir(f'{obj}_graphs/layers_{min_layer}_to_{max_layer}')
if not os.path.isdir(f'{obj}_adjacencies'):
    os.mkdir(f'{obj}_adjacencies')
if not os.path.isdir(f'{obj}_adjacencies/layers_{min_layer}_to_{max_layer}'):
    os.mkdir(f'{obj}_adjacencies/layers_{min_layer}_to_{max_layer}')

for file in tqdm(files):
    if file.startswith('.'):
        continue
    with open(f'{obj}_graphs/layers_{min_layer}_to_{max_layer}/{file}','rb') as f:
        vertices, simplices, boundary = pkl.load(f)
    vertices, simplices, boundary = torch.tensor(vertices,dtype=torch.float32), torch.tensor(simplices,dtype=torch.int32), torch.tensor(boundary,dtype=torch.float32)
        
    edges = torch.zeros((len(vertices), len(vertices)))
    for simplex in simplices:
        for i,v in enumerate(simplex):
            for w in simplex[i+1:]:
                edges[v, w] = 1
                edges[w, v] = 1
    
    adjacency = edges.to_sparse()
    
    boundary_points = [np.argmin(np.linalg.norm(vertices-boundary_point,axis=1)) for boundary_point in boundary]
    boundary_points = list(set(boundary_points))
    boundary_tensor = torch.sparse_coo_tensor([boundary_points], [1 for _ in boundary_points],(len(vertices),), dtype=torch.float32)
    boundary_tensor = boundary_tensor.to_dense()
    
    bottom_z, top_z = vertices[:,-1].min(), vertices[:,-1].max()
    bottom_boundary = torch.where(vertices[:,-1]<=bottom_z,1.,0.)
    top_boundary = torch.where(vertices[:,-1]>=top_z,1.,0.)
    side_boundary = (boundary_tensor - bottom_boundary - top_boundary).clip(0,1)
    
    distances = [torch.linalg.norm(vertices[i]-vertices[j]) for i,j in zip(*adjacency.indices())]
    distance_t = torch.sparse_coo_tensor(adjacency.indices(),distances,adjacency.shape).coalesce()
    with open(f'{obj}_adjacencies/layers_{min_layer}_to_{max_layer}/{file}','wb') as f:
        pkl.dump((vertices,distance_t,bottom_boundary,top_boundary,side_boundary), f)
    