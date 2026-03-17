import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle as pkl
import os
from copy import deepcopy
import time

device = 'cpu'

class ConnectivityModel(nn.Module):
    """
    Model for determining the connectivity for the heat flow along the (directed) edges in the graph ('\varphi')
    """
    
    def __init__(self, hidden_dims=[256,],loss_weights=[1.,1.,.1,.1],device=device):
        super().__init__()
        
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float32,device=device) # weights for the regularization loss functions
        
        self.temp_regulariser = nn.Parameter(torch.tensor(1e-3,dtype=torch.float32, device=device)) # multiplier for the input temperature values
        
        self.layer_1 = nn.Linear(12,hidden_dims[0],device=device)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i],hidden_dims[i+1],device=device) for i in range(len(hidden_dims)-1)])
        self.final_layer = nn.Linear(hidden_dims[-1],1,device=device)
        
    def forward(self, t1, t2, c1, c2, dens1, dens2, dist, fit=False, eps=1e-9):
        
        if fit: # enable gradients for the inputs in order to calculate the regularization losses
            t1 = t1.detach()
            t1.requires_grad=True
            t2 = t2.detach()
            t2.requires_grad=True
            dens1 = dens1.detach()
            dens1.requires_grad=True
            dens2 = dens2.detach()
            dens2.requires_grad=True
            dist = dist.detach()
            dist.requires_grad=True
            
        # normalization of the input temperatures
        t1 = torch.tanh(self.temp_regulariser * t1)
        t2 = torch.tanh(self.temp_regulariser * t2)
        
        # concatenate the argument tensors
        args = torch.cat([t1.unsqueeze(-1),t2.unsqueeze(-1),c1,c2,dist.unsqueeze(-1),(t2-t1).unsqueeze(-1),dens1.unsqueeze(-1),dens2.unsqueeze(-1)], dim=-1)
        
        # evaluate the model
        x = torch.tanh(self.layer_1(args))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = self.final_layer(x)
        x = F.softplus(x)
        
        if fit: # calculate the regularization losses
            grad = torch.autograd.grad([x[i] for i in range(len(x))],[t1,t2,dens1,dens2,dist],retain_graph=True)
            dist_loss = ((grad[-1]/(x.squeeze(-1)+eps))-(-2/(dist+eps))).square().mean()
            temp_loss = grad[0].square().mean() + grad[1].square().mean()
            dens_loss = ((grad[2]/(x.squeeze(-1)+eps))-(1/(dens1+eps))).square().mean() + ((grad[3]/(x.squeeze()+eps))-(-1/(dens2+eps))).square().mean()
            symmetry_loss = (x-self(t2,t1,c2,c1,dens1,dens2,dist)).square().mean()
            conn_loss = torch.dot(torch.stack([dist_loss,temp_loss,dens_loss,symmetry_loss]),self.loss_weights)
            return x,conn_loss
        return x


class DissipationModel(nn.Module):
    """
    Model for estimating the heat dissipation at each vertex ('\psi')
    """
    
    def __init__(self,hidden_dims=[256,], boundary_temp=124.9,device=device):
        super().__init__()
        
        self.temp_regulariser = nn.Parameter(torch.tensor(1e-2,dtype=torch.float32, device=device)) # multiplier for the input temperature values
        
        self.layer_1 = nn.Linear(5,hidden_dims[0],device=device)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i],hidden_dims[i+1],device=device) for i in range(len(hidden_dims)-1)])
        self.final_layer = nn.Linear(hidden_dims[-1],1,device=device)
        
        self.coefs = nn.Parameter(-torch.ones(3,dtype=torch.float32, device=device)) # learnable parameters in the regularization loss 
        
    def forward(self,temperature, classes, density, fit=False):
        
        if fit: # enable gradients for the inputs in order to calculate the regularization loss
            temperature = temperature.detach()
            temperature.requires_grad=True
            density = density.detach()
            density.requires_grad=True
            
        temperature = torch.tanh(self.temp_regulariser * temperature) # normalization of the input temperatures
        
        # concatenate the argument tensors
        args = torch.cat([temperature.unsqueeze(-1), classes, density.unsqueeze(-1)], dim=-1)
        
        # evaluate the model
        x = torch.tanh(self.layer_1(args))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = self.final_layer(x)
        
        if fit: # calculate the regularization losses
            grad = torch.autograd.grad([x[i] for i in range(len(x))],[temperature, density],retain_graph=True)
            bottom_ix = torch.where(classes[:,0]==1,True,False)
            top_ix = torch.where(classes[:,1]==1,True,False)
            side_ix = torch.where(classes[:,2]==1,True,False)
            interior_ix = torch.where(classes.sum(dim=1)==0,True,False)
            temp_loss = (grad[0] - self.coefs[0]).where(bottom_ix,torch.zeros_like(grad[0])).square().mean() + \
                        (grad[0] - self.coefs[1]).where(top_ix,torch.zeros_like(grad[0])).square().mean() + \
                        (grad[0] - self.coefs[2]).where(side_ix,torch.zeros_like(grad[0])).square().mean() + \
                        grad[0].where(interior_ix,torch.zeros_like(grad[0])).square().mean()
            dens_loss = grad[1].square().mean()
            return x, temp_loss + dens_loss
        return x
    
    
# class LaserModel(nn.Module):
#     """
#     Model for the temperature change caused by the laser
#     """
#     def __init__(self,device=device):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(2,64),
#             nn.Tanh(),
#             nn.Linear(64,1),
#             nn.LeakyReLU(.05), # outputs should be nonnegative, but the model might 'die', if relu was used
#         ).to(device)
#         
#     def forward(self,temperatures,distances):
#         return self.model(torch.stack([temperatures/1000,distances/100],dim=-1))
    
class LaserModel(nn.Module):
    """
    Model for the temperature change caused by the laser
    """
    def __init__(self,device=device):
        super().__init__()
        self.intensity = nn.Parameter(50*torch.ones(1))
        self.decay = nn.Parameter(.4*torch.ones(1))
        
    def forward(self,distances):
        return self.intensity * torch.exp(-(self.decay*distances).square())
    
    
class CNModel(nn.Module):
    def __init__(self, k=4, boundary_value=124.9,device=device):
        super().__init__()
        self.boundary_value = boundary_value
        self.k = k
        
        # initialize the submodels
        self.conn_model = ConnectivityModel(device=device)
        self.diss_model = DissipationModel(device=device)
        self.laser_model = LaserModel(device=device)
    
    
class DiffusionModel(nn.Module):
    def __init__(self, k=5, boundary_value=124.9,device=device):
        super().__init__()
        self.boundary_value = boundary_value
        self.k = k
        
        # initialize the submodels
        self.conn_model = ConnectivityModel(device=device)
        self.diss_model = DissipationModel(device=device)
        self.laser_model = LaserModel(device=device)
        
        # weights of the convolution model ('\Theta_j')
        self.weights = nn.Parameter(1e-2*torch.tensor([1/np.math.factorial(j) for j in range(1,k+1)],dtype=torch.float32, device=device))
        
    def forward(self, distance_adj, densities, vertex_class, temperature, dt, laser_dist=None, fit=False, eps=1e-9):
        
        if laser_dist is None: # if no laser is present, assign a very large distance to the laser position for each vertex
            laser_dist = 1e9 * torch.ones_like(temperature)
        
        # prepare arguments for the submodels
        c1 = vertex_class[distance_adj.indices()[0]]
        c2 = vertex_class[distance_adj.indices()[1]]
        
        dens1 = densities[distance_adj.indices()[0]]
        dens2 = densities[distance_adj.indices()[1]]
        
        dists = distance_adj.values()
        
        t1 = torch.index_select(temperature,0,distance_adj.indices()[0]) - self.boundary_value
        t2 = torch.index_select(temperature,0,distance_adj.indices()[1]) - self.boundary_value
        
        # determine the edge connectivity for the current state 
        if fit:
            connectivity, conn_loss = self.conn_model(t1,t2,c1,c2,dens1,dens2,dists,fit=True)
        else:
            connectivity = self.conn_model(t1,t2,c1,c2,dens1,dens2,dists)
        
        # create the graph laplacian from the edge connectivities
        connectivity = connectivity.squeeze(-1)
        conn_matrix = torch.sparse_coo_tensor(distance_adj.indices(),connectivity,distance_adj.shape)
        conn_matrix = conn_matrix.coalesce()
        degree = torch.sparse.sum(conn_matrix,dim=1)
        degree_matrix = torch.sparse_coo_tensor(torch.stack([degree.indices()[0],degree.indices()[0]],dim=0), degree.values(), conn_matrix.shape)
        laplacian = conn_matrix - degree_matrix
        
        # evaluate the dissipation model
        if fit:
            diss_vector, diss_loss = self.diss_model(temperature.view(-1),vertex_class,densities,fit=True)
        else:
            diss_vector = self.diss_model(temperature.view(-1),vertex_class,densities)
        
        # calculate the products (dt*L)^j @ T, j=0,...,k by consecutive left-multiplication with dt*L
        summands = [temperature.unsqueeze(-1) + .5 * dt * diss_vector,]
        for _ in range(self.k):
            summands.append(dt*torch.sparse.mm(laplacian, summands[-1]).to_dense())
        
        # predict the new temperature state as the weighted sum of the (dt*L)^j @ T, using the model weights
        sum_mat = torch.cat(summands[1:],dim=1)
        new_temp = temperature + sum_mat @ self.weights + dt * diss_vector.squeeze(-1)
        
        # predict and add the temperature change caused by the laser
        laser_heat = self.laser_model(laser_dist).squeeze(-1)
        new_temp = new_temp + dt * laser_heat
        
        if fit: # return the predicted value, together with the values for the regularizing loss functions
            return new_temp, conn_loss, diss_loss, dt * diss_vector, dt * laser_heat
        return new_temp
    
    
    def develop(self,X,initial_state=None):
        """
        use the model to update the internal heat state of the part, using an initial state and taking surface data into account
        """
        
        if initial_state is not None:
            state = initial_state.detach().clone()
        else:
            state = self.boundary_value * torch.ones((X[0][0].shape[0],),dtype=torch.float32)
        
        for distances, densities, boundary, surface_temp, time_delta in X:
            state[surface_temp.indices()] = 0.
            state = state + surface_temp.to_dense()
            pred = self(distances, densities, boundary, state, time_delta)
            state = pred.detach().view(-1)
        return state
    
    
    def save(self,path,compiled=False, override=False):
        if os.path.isfile(path) and not override:
            raise ValueError('file already exists!')
        if compiled:
            torch.save(deepcopy(self),path)
        else:
            model_state = {
                'state_dt': deepcopy(self.state_dict()),
                'boundary_value': self.boundary_value
            }
            torch.save(model_state,path)
    
    @staticmethod
    def load(path, compiled=False):
        if compiled:
            loaded_model = torch.load(path)
            return loaded_model
        model = DiffusionModel()
        model_state = torch.load(path)
        model.load_state_dict(model_state['state_dt'])
        model.boundary_value = model_state['boundary_value']
        return model
    
inv_ = np.vectorize(lambda x: 1/x if not x == 0. else 0.)

def scale_invariant_density(space,return_avg_dist=False):
    """
    calculate the scale-invariant-density, as defined in: https://doi.org/10.48550/arXiv.2110.01286
    with adaptation for 3d data
    """
    
    ret_val = None
    dim = space.shape[-1]
    if dim != 2 and dim != 3:
        print(space.shape)
        raise NotImplementedError()
    pairings = np.tile(space,(space.shape[0],1,1)) - np.tile(space,(space.shape[0],1,1)).transpose([1,0,2])
    dens = np.sum(np.square(pairings),axis=-1)
    if dim == 2:
        ret_val = np.sum(inv_(np.sqrt(dens)), axis=1)
    else:
        ret_val = np.sum(inv_(dens), axis=1)
    
    if return_avg_dist:
        return ret_val, np.linalg.norm(pairings.reshape(-1,dim),axis=-1).mean()
    return ret_val

def load_data_layer(layer,device=device):
    """
    load the previously created graphs and the extracted surface data of the given layer
    """
    
    with open(f'pyramid_4/layer_{571+layer}.pkl','rb') as f:
        layer_data = pkl.load(f)
    with open(f'pyramid_4_adjacencies/layers_571_to_1063/layer_{layer}.pkl','rb') as f:
        graph_data = pkl.load(f)
    with open(f'pyramid_4_graphs/layers_571_to_1063/layer_{layer}.pkl','rb') as f:
        simplex_data = pkl.load(f)

    timestamps, temperatures, laser_position = layer_data
    times_ms = [(ts.asm8.astype('int')/1e6) for ts in timestamps]
    vertices, distance_t, bottom_boundary, top_boundary, side_boundary = graph_data
    distance_t = distance_t.coalesce()
    simplices = simplex_data[1]
    
    top_layer_indices = top_boundary.to_sparse().indices()[0]
    top_layer_vertices = vertices[top_layer_indices][:,:2].numpy().astype(int)

    top_layer_x, top_layer_y = list(zip(*top_layer_vertices))
    top_layer_x, top_layer_y = torch.tensor(top_layer_x), torch.tensor(top_layer_y)
    
    temp_values = torch.tensor(np.array([temp_image[top_layer_x,top_layer_y] for temp_image in temperatures]), dtype=torch.float32,device=device)
    temp_vecs = [torch.sparse_coo_tensor(top_layer_indices.unsqueeze(0), temp_vals, (len(vertices),), dtype=torch.float32,device=device).coalesce() for temp_vals in temp_values]
    
    boundary_masks = torch.stack([bottom_boundary, top_boundary, side_boundary],dim=-1)
    
    densities = torch.tensor(scale_invariant_density(vertices), dtype=torch.float32)
    densities = densities/densities.mean()
    
    laser_distance = []
    for laser_pos in laser_position:
        if laser_pos is None:
            laser_distance.append(1e9 * torch.ones(len(vertices)))
        else:
            top_layer_z = vertices[:,2].max()
            laser_distance.append(torch.tensor([torch.linalg.norm(vertex[:2]-laser_pos) if vertex[2] == top_layer_z else 1e9 for vertex in vertices], dtype=torch.float32, device=device))
    
    return vertices.to(device), distance_t.to(device), densities.to(device), boundary_masks.to(device), times_ms, temp_vecs, laser_distance, simplices


def load_surface_temperatures(layer):
    with open(f'pyramid_7/layer_{571+layer}.pkl','rb') as f:
        layer_data = pkl.load(f)
    return layer_data[1]