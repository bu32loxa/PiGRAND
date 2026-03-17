import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DiffusionModel import *
from ImplicitModel import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
import os
from copy import deepcopy
import time
import datetime as dt
from sklearn.random_projection import GaussianRandomProjection
from scipy.spatial import Delaunay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_state_up(vertices, state, simplices, vmin=300, vmax=900, save_path=None, show=True, plot_format='pdf', set_limits=True, backend=None, C_to_K=True, bounds=None):
    """
    Plot the predicted temperature state of the part, removing extraneous patches by restricting to a 3D bounding box.
    Args:
        vertices (array): Array of vertex coordinates.
        state (array): Array of state values corresponding to each vertex.
        simplices (array): Array of simplices (triangles or tetrahedra).
        vmin (float): Minimum state value for the color map.
        vmax (float): Maximum state value for the color map.
        bounds (dict): Dictionary defining x, y, and z bounds as {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}.
    """
    if backend is not None:
        mpl.use('Agg')

    if C_to_K:
        state = state + 273.15

    fig = plt.figure(figsize=(8, 7))

    if len(simplices[0]) == 3:
        triangles = simplices
    else:
        triangles = np.concatenate([
            simplices[:, (0, 1, 2)],
            simplices[:, (0, 1, 3)],
            simplices[:, (0, 2, 3)],
            simplices[:, (1, 2, 3)],
        ])

    # Apply bounding box filtering
    if bounds is not None:
        x_bounds, y_bounds, z_bounds = bounds['x'], bounds['y'], bounds['z']
        valid_indices = np.where(
            (vertices[:, 0] >= x_bounds[0]) & (vertices[:, 0] <= x_bounds[1]) &
            (vertices[:, 1] >= y_bounds[0]) & (vertices[:, 1] <= y_bounds[1]) &
            (vertices[:, 2] >= z_bounds[0]) & (vertices[:, 2] <= z_bounds[1])
        )[0]

        # Filter vertices and triangles
        valid_triangles = [tri for tri in triangles if all(v in valid_indices for v in tri)]
        triangles = np.array(valid_triangles)

    # Filter triangles based on state values
    triangle_states = np.array([state[simplex.tolist()].mean() for simplex in triangles])
    valid_triangles = triangles[triangle_states > vmin]  # Keep triangles with average state > vmin

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    colors = np.array([state[simplex.tolist()].mean() for simplex in valid_triangles])
    plot = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=valid_triangles, cmap='jet', alpha=.1, vmin=vmin, vmax=vmax)
    plot.set_array(colors)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if set_limits:
        ax.set_xlim(bounds['x'] if bounds else (0, 25))
        ax.set_ylim(bounds['y'] if bounds else (0, 25))
        ax.set_zlim(bounds['z'] if bounds else (0, 15))
    else:
        ax.set_aspect('auto', adjustable='box')
    ax.view_init(20, 160)

    cb_ax = fig.add_axes([.91, .25, .015, .5])
    cbar = fig.colorbar(plot, orientation='vertical', cax=cb_ax)
    cbar.set_label('T[K]')
    cbar.set_alpha(1)
    cbar.draw_all()

    if save_path is not None:
        if plot_format == 'png':
            plt.savefig(save_path, bbox_inches='tight', format=plot_format, dpi=200)
        else:
            plt.savefig(save_path, bbox_inches='tight', format=plot_format)
    if show:
        plt.show()
    plt.clf()
    plt.close('all')

def plot_state(vertices,state,simplices,vmin=300,vmax=900,save_path=None,show=True,plot_format='pdf',set_limits=True,backend=None,C_to_K=True):
    """
    plot the predicted temperature state of the part
    """
    if backend is not None:
        mpl.use('Agg')

    if C_to_K:
        state = state + 273.15
    fig = plt.figure(figsize=(8, 7))
    
    if len(simplices[0]) == 3:
        triangles = simplices
    else:
        triangles = np.concatenate([
                    simplices[:, (0, 1, 2)],
                    simplices[:, (0, 1, 3)],
                    simplices[:, (0, 2, 3)],
                    simplices[:, (1, 2, 3)],
        ])
    
    ax = fig.add_subplot(1,1,1,projection='3d')
    colors = np.array([state[simplex.tolist()].mean() for simplex in triangles])

    """# Add this before plot_trisurf
    centroids = vertices[triangles].mean(axis=1)  # centroid of each triangle

    # Compute the centroid of the whole mesh
    mesh_center = vertices.mean(axis=0)

    # Compute distances of each triangle's centroid to the overall mesh center
    distances = np.linalg.norm(centroids - mesh_center, axis=1)

    # Filter out triangles with too large distance (adjust threshold as needed)
    distance_threshold = 20  # <-- tune this value as needed
    valid_mask = distances < distance_threshold
    triangles = triangles[valid_mask]

    # Compute colors for the filtered triangles
    colors = np.array([state[simplex].mean() for simplex in triangles])"""

    # Now plot using facecolors
    plot = ax.plot_trisurf(vertices[:,0],vertices[:,1],vertices[:,2], triangles=triangles,cmap='jet', alpha=.1, vmin=vmin, vmax=vmax) # vmin, vmax
    plot.set_array(colors)

    #plot.autoscale()
    #ax.xaxis.set_ticklabels([])
    #ax.yaxis.set_ticklabels([])
    #ax.zaxis.set_ticklabels([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if set_limits:
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_zlim(0, 15)
    else:
        ax.set_aspect('auto', adjustable='box')
    ax.view_init(20, 160)

    
    # ax = fig.add_subplot(1,2,2,projection='3d')
    # verts = ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2],c=state,cmap='jet',vmin=vmin,vmax=vmax)
    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])

    cb_ax = fig.add_axes([.91,.25,.015,.5])
    cbar = fig.colorbar(plot, orientation='vertical', cax=cb_ax)
    cbar.set_label('T[K]')
    cbar.set_alpha(1)
    cbar.draw_all()
    if save_path is not None:
        #pickle_path = '.'.join(save_path.split('.')[:-1])+'.pkl'
        #with open(pickle_path,'wb') as f:
            #pkl.dump(fig,f)
        if plot_format == 'png':
            plt.savefig(save_path, bbox_inches='tight', format=plot_format, dpi=200)
        else:
            plt.savefig(save_path, bbox_inches='tight', format=plot_format)
    if show:
        plt.show()
    plt.clf()
    plt.close('all')
    
def plot_surface(surface_data,vmin=300,vmax=3000,save_path=None, show=True, set_limits=True,C_to_K=True):
    """
    plot surface data from thermal images
    """
    if C_to_K:
        surface_data = surface_data + 273.15
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)
    image = ax.imshow(np.flip(surface_data,axis=1), interpolation='spline36', cmap='jet',
                       origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    if not set_limits:
        ax.set_aspect('equal', adjustable='box')
    if save_path is not None:
        pickle_path = '.'.join(save_path.split('.')[:-1])+'.pkl'
        with open(pickle_path, 'wb') as f:
            pkl.dump(fig,f)
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    if show:
        plt.show()
    plt.close()

    
def plot_surface_state(vertices,state, vmin=300, vmax=750, save_path=None, show=False, set_limits=True,vertex_multipliers=None,C_to_K=True):
    """
    plot predicted surface temperature
    """
    if C_to_K:
        state = state + 273.15

    
    surface_z = vertices[:, 2].max()
    surface_state = [(vertex[:2], temperature) for vertex, temperature in zip(vertices, state) if vertex[2] == surface_z]
    vertices, temperatures = zip(*surface_state)
    triangles = Delaunay(vertices)
    vertices, temperatures = list(vertices), list(temperatures)


    
    vertices = np.flip(vertices, axis=1)
    
    vertices[:, 0] = 30-vertices[:, 0]
    
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(1, 1, 1)
    
    if vertex_multipliers is not None:
        vertices[:, 0] *= vertex_multipliers[0]
        vertices[:, 1] *= vertex_multipliers[1]

    #cmap = mpl.colormaps['turbo']
    triangulation = mpl.tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles.simplices)
    h = ax.tripcolor(triangulation, temperatures, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
    # ax.scatter(vertices[:,0], vertices[:,1], c=temperatures, cmap='turbo', vmin=vmin, vmax=vmax)
    if set_limits:
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
    else:
        ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_ticklabels([])
    #ax.yaxis.set_ticklabels([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    K = fig.colorbar(h, cax=cax)
    K.set_label('T[K]', fontsize=15)
    ax.set_title('GRAND', fontsize=15)
    if save_path is not None:
        pickle_path = '.'.join(save_path.split('.')[:-1])+'.pkl'
        with open(pickle_path, 'wb') as f:
            pkl.dump(fig, f)
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    if show:
        plt.show()
    plt.close()


def plot_surface_state_error(vertices, state, y, vmin=0, vmax=50, save_path=None, show=False, set_limits=True,
                       vertex_multipliers=None, C_to_K=True):
    """
    plot predicted surface temperature
    """
    if C_to_K:
        state = state + 273.15
        y = y + 273.15

    surface_z = vertices[:, 2].max()
    surface_state = [(vertex[:2], temperature) for vertex, temperature in zip(vertices, state) if
                     vertex[2] == surface_z]
    vertices, temperatures = zip(*surface_state)
    triangles = Delaunay(vertices)
    vertices, temperatures = list(vertices), list(temperatures)
    y_temp = y.values().view(1,-1)
    ende = len(temperatures)
    for i in range(0, ende):
        temperatures[i] = np.abs(temperatures[i] - y_temp[0][i])

    vertices = np.flip(vertices, axis=1)

    vertices[:, 0] = 30 - vertices[:, 0]

    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(1, 1, 1)

    if vertex_multipliers is not None:
        vertices[:, 0] *= vertex_multipliers[0]
        vertices[:, 1] *= vertex_multipliers[1]

    # cmap = mpl.colormaps['turbo']
    triangulation = mpl.tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles.simplices)
    h = ax.tripcolor(triangulation, temperatures, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
    # ax.scatter(vertices[:,0], vertices[:,1], c=temperatures, cmap='turbo', vmin=vmin, vmax=vmax)
    if set_limits:
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
    else:
        ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    #ax.xaxis.set_ticklabels([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    K = fig.colorbar(h, cax=cax)
    K.set_label('$\epsilon_{abs}$', fontsize=15)
    ax.set_title('PiGRAND', fontsize=15)
    if save_path is not None:
        pickle_path = '.'.join(save_path.split('.')[:-1]) + '.pkl'
        with open(pickle_path, 'wb') as f:
            pkl.dump(fig, f)
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    if show:
        plt.show()
    plt.close()


def plot_diff(vertices, state, simplices, save_path=None, show=True):
    """
    plot the predicted temperature state of the part
    """
    maxabs = np.max(np.abs(state))
    
    fig = plt.figure(figsize=(8,7))
    
    if len(simplices[0]) == 3:
        triangles = simplices
    else:
        triangles = np.concatenate([
                    simplices[:,(0,1,2)],
                    simplices[:,(0,1,3)],
                    simplices[:,(0,2,3)],
                    simplices[:,(1,2,3)],
        ])
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    colors = np.array([state[simplex.tolist()].mean() for simplex in triangles])
    plot = ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=triangles,cmap='jet', alpha=.1,vmin=-200,vmax=150)
    plot.set_array(colors)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlim(0,30)
    ax.set_ylim(0,30)
    ax.set_zlim(0,15)
    ax.view_init(20, 160)

    
    # ax = fig.add_subplot(1,2,2,projection='3d')
    # verts = ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2],c=state,cmap='jet',vmin=vmin,vmax=vmax)
    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])
    
    cb_ax = fig.add_axes([.91,.25,.015,.5])
    cbar = fig.colorbar(plot,orientation='vertical',cax=cb_ax)
    cbar.set_label('T[K]')
    cbar.set_alpha(1)
    cbar.draw_all()
    if save_path is not None:
        pickle_path = '.'.join(save_path.split('.')[:-1])+'.pkl'
        with open(pickle_path,'wb') as f:
            pkl.dump(fig,f)
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    if show:
        plt.show()
    plt.close()


def plot_surface_diff(vertices,state, save_path=None, show=True):
    """
    plot predicted surface temperature
    """
    
    maxabs = np.max(np.abs(state))
    
    surface_z = vertices[:,2].max()
    surface_state = [(vertex[:2],temperature) for vertex,temperature in zip(vertices,state) if vertex[2] == surface_z]
    vertices, temperatures = zip(*surface_state)
    triangles = Delaunay(vertices)
    vertices, temperatures = list(vertices), list(temperatures)

    maxabs = np.max(np.abs(temperatures))
    
    vertices = np.flip(vertices, axis=1)
    
    vertices[:, 0] = 30-vertices[:,0]
    
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(1, 1, 1)
    
    #cmap = mpl.colormaps['turbo']
    triangulation = mpl.tri.Triangulation(vertices[:,0], vertices[:,1], triangles.simplices)
    plot = ax.tripcolor(triangulation, temperatures, shading='gouraud', cmap='BrBG', vmin=-maxabs, vmax=maxabs)
    # ax.scatter(vertices[:,0], vertices[:,1], c=temperatures, cmap='turbo', vmin=vmin, vmax=vmax)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    cb_ax = fig.add_axes([.91,.25,.015,.5])
    cbar = fig.colorbar(plot,orientation='vertical',cax=cb_ax)    
    cbar.set_alpha(1)
    cbar.draw_all()
    if save_path is not None:
        pickle_path = '.'.join(save_path.split('.')[:-1])+'.pkl'
        with open(pickle_path,'wb') as f:
            pkl.dump(fig,f)
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    if show:
        plt.show()
    plt.close()



def transfer_state(vertices_source, state_source, vertices_target, space_dim=(100,100,100000)):
    """
    transfer the temperatures in the source graph to the target vertices at the same spatial location 
    (for passing from the graph for one layer to the graph for the next layer)
    """
    
    device = state_source.device
    indices = torch.stack([vertices_source[:,0].type(torch.int32),
                           vertices_source[:,1].type(torch.int32),
                           (1000*vertices_source[:,2]).type(torch.int32)])

    space_temperatures = torch.sparse_coo_tensor(indices,state_source.to_dense(),space_dim,device=device)
    state_target = torch.stack([space_temperatures[int(x),int(y),int(1000*z)] for x,y,z in vertices_target])
    return state_target


def calculate_energy(distribution, density, classes):
    """
    calculate the potential energy of the heat state (i.e. the second moment of the temperature over the domain)
    """
    
    boundary_order = classes.sum(dim=1)
    weights = (1/density)*torch.pow(2.,-boundary_order)
    weights = weights/weights.sum()
    energy = torch.dot(distribution.square(), weights) - torch.dot(distribution, weights).square()
    return energy

def calculate_heat(distribution, density, classes, dissipation=None):
    """
    calculate the total thermal energy 
    """
    
    boundary_order = classes.sum(dim=1)
    weights = (1/density)*torch.pow(2.,-boundary_order)
    heat = torch.dot(distribution,weights)
    if dissipation is None:
        return heat
    dissipation_heat = torch.dot(dissipation,weights)
    return heat, dissipation_heat
    
def fit_model(model, 
              depths_iters=[(2, 250), (5, 100), (10, 50), (25, 20), (50, 10), (100, 5), (250, 3), (500, 2)], #pairs of training depth and number of training iterations for the depth
              initial_state=None,
              lambd=1, 
              alpha_conn=1., 
              alpha_diss=1., 
              alpha_energy= 1., 
              alpha_heat= 1., 
              alpha_max_min= 1.,
              t_max=1e3,
              lr=1e-5,
              betas=(.5,.999),
              save_best=False,
              save_path='diffusion_model_implicit.pt',
              obj = 'pyramid_7_m',
              min_layer = 571,
              max_layer = 1076,
              vertex_multipliers = None,
              set_plot_limits = True,
              whole_layer = False,
              device=device):
    
    n_layers = depths_iters[-1][0] # total number of layers in the final part
    
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    loss_fn = nn.MSELoss()
    
    full_iters = depths_iters[-1][1]
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1., end_factor=1./full_iters, total_iters=full_iters)
    
    history = []
    energy_loss =[]
    start_time = int(time.time())
    
    save_depth = False
    
    # increase the number of layers used for training, as later layers are sensitive to the hidden state obtained from the model predictions, and
    # thus require a relatively well-trained model in order to produce reasonable predictions
    for dix, (depth, depth_iterations) in enumerate(depths_iters):
        print('depth:', depth)
        
        if dix==(len(depths_iters)-1):
            save_depth = True
            best_loss = np.inf
        
        if not whole_layer and depth >= 5:
            ditr = range(depth_iterations)
            pbar=True
        else:
            ditr = tqdm(range(depth_iterations))
            pbar=False

        for depth_iter in ditr:
            
            # initialize values for accumulating the losses for the current training depth
            depth_avg_loss = 0.
            depth_fwd_loss = 0.
                
            if pbar:
                itr = tqdm(range(depth))
            else:
                itr = range(depth)
                
            for layer in itr:

                data = load_data_layer(layer, obj=obj, layers=f'{min_layer}_to_{max_layer}', min_layer=min_layer, max_layer=max_layer, vertex_multipliers=vertex_multipliers, device=device)
                
                #initialize values for accumulating losses for the current layer
                layer_loss = 0.
                layer_const_loss = 0.
                layer_conn_loss = 0.
                layer_diss_loss = 0.
                layer_energy = 0.
                layer_avg_steps = 0.

                if layer == 0:
                    if initial_state is not None:
                        layer_init_state = initial_state
                    else:
                        layer_init_state = model.boundary_value * torch.ones((data[0].shape[0],),dtype=torch.float32,device=device)
                
                # determine the initial heat state for the layer by transfering the final state of the previous layer, taking into account the
                # diffusion in the time passed during recoating    
                else: 
                    time_delta = (data[4][0] - prev_last_time)/1000
                    state = model(prev_distances, prev_densities, prev_boundary, state, time_delta).detach().view(-1)
                    layer_init_state = transfer_state(prev_vertices,state,data[0])
                
                
                prev_vertices = data[0]
                prev_distances = data[1]
                prev_densities = data[2]
                prev_boundary = data[3]
                prev_last_time = data[4][-1]

                state = layer_init_state
                
                # sample the evaluation time-steps (either random or n-step)
                
                # jumps = np.random.poisson(lambd,len(data[5])) + 1
                # eval_ixs = np.cumsum(jumps)
                # eval_ixs = eval_ixs[np.where(eval_ixs<len(data[5]))].tolist()
                # if eval_ixs[0] > 0:
                #     eval_ixs = [0,] + eval_ixs
                
                if whole_layer:
                    eval_ixs = [0,len(data[5])-1]
                else:
                    eval_ixs = np.arange(0,len(data[5]),lambd).tolist()
                    if eval_ixs[-1] < len(data[5])-1:
                        eval_ixs.append(len(data[5])-1)
                
                X = [(data[1], data[2], data[3], data[5][eval_ixs[i]], (data[4][eval_ixs[i+1]]-data[4][eval_ixs[i]])/1000, data[6][eval_ixs[i+1]]) for i in range(len(eval_ixs)-1)]
                Y = [data[5][eval_ixs[i]] for i in range(1,len(eval_ixs))]
                
                for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X,Y):
                    
                    optim.zero_grad()
                    
                    #assign the surface temperature values
                    state[surface_temp.indices()[0]] = 0.
                    state = state + surface_temp.to_dense()
                    
                    # evaluate the model and loss function
                    pred, conn_loss, diss_loss, diss_vec, laser_heat = model(distances.to(device), densities.to(device), boundary.to(device), state.to(device), time_delta.to(device), laser_dist.to(device), fit=True)
                    pred_loss = loss_fn(pred[y.indices()].view(-1, 1), y.values().view(-1, 1))
                    #print(laser_dist[0])
                    
                    # calculate the regularizing loss functions
                    pred = pred - laser_heat
                    neighbor_temp = (torch.sign(distances).to_dense() * pred.view(-1).to_dense())
                    max_principle_violation = (torch.relu(pred.view(-1)-neighbor_temp.max(dim=1).values)).where(torch.logical_and(pred>state,boundary.sum(dim=1)==0),torch.zeros_like(pred)).square().mean()
                    min_principle_violation = (torch.relu(neighbor_temp.min(dim=1).values-pred.view(-1))).where(torch.logical_and(pred>state,boundary.sum(dim=1)==0),torch.zeros_like(pred)).square().mean()
                    energy_violation = torch.relu(
                                calculate_energy(pred, densities,boundary) # energy of prediction
                               -calculate_energy(state, densities,boundary) # energy of previous state
                             )
                    heat, heat_diss = calculate_heat(pred, densities,boundary, dissipation=diss_vec.squeeze())
                    total_heat_diff = ((heat-heat_diss)/state.sum()-1).square()
                    
                    # assemble the total loss
                    loss = pred_loss +\
                           alpha_max_min * (max_principle_violation + min_principle_violation) +\
                           alpha_energy * energy_violation +\
                           alpha_conn * conn_loss + \
                           alpha_diss * diss_loss + \
                           alpha_heat * total_heat_diff
                # optimization step
                    loss.backward()
                    optim.step()    
                    
                    # log the loss values
                    push_fwd_loss = loss_fn(state[y.indices()].view(-1,1),y.values().view(-1,1))
                    layer_loss += pred_loss.item()/(len(eval_ixs)-1)
                    layer_const_loss += push_fwd_loss.item()/(len(eval_ixs)-1)
                    layer_conn_loss += conn_loss.item()/(len(eval_ixs)-1)
                    layer_diss_loss += diss_loss.item()/(len(eval_ixs)-1)
                    layer_energy += (energy_violation + total_heat_diff + max_principle_violation + min_principle_violation).item()/(len(eval_ixs)-1)

                    state = pred.detach().clip(0, t_max).view(-1)

                layer_avg_steps += len(eval_ixs)-1

                if pbar:
                    if layer_const_loss != 0.:
                        itr.set_postfix({'l_T':layer_loss,
                                        'rel':(layer_const_loss-layer_loss)/layer_const_loss,
                                         'l_conn':layer_conn_loss,
                                         'l_diss':layer_diss_loss,
                                        'l_reg':(layer_conn_loss+layer_diss_loss),
                                        'l_E': layer_energy,
                                        't_min':state.min().item(), 
                                        't_max':state.max().item()})

                #energy_loss.append(layer_energy)
                depth_avg_loss += layer_loss
                depth_fwd_loss += layer_const_loss
            if not depth_fwd_loss == 0:
                if pbar:
                    print(depth_avg_loss/depth, (depth_fwd_loss-depth_avg_loss)/depth_fwd_loss)
                else:
                    ditr.set_postfix({'l_T':depth_avg_loss/depth, 'rel':(depth_fwd_loss-depth_avg_loss)/depth_fwd_loss})
            if save_depth:
                scheduler.step()
                if depth_avg_loss < best_loss:
                    best_loss = depth_avg_loss
                    print('saving model state.')
                    model.save(save_path, compiled=True, override=True)

        #plot_state(data[0].detach().numpy(), state.detach().numpy(),data[7], set_limits=set_plot_limits, save_path=f'plots/depth_{depth}_plot{dix}.svg')
        print('parameters:')
        print(model.diss_model.coefs)
        print(model.laser_model.intensity.item(),', ', model.laser_model.decay.item(), sep='')

    return history


def fit_model_batch_diffusion(model,
              depths_iters=[(2, 250), (5, 100), (10, 50), (25, 20), (50, 10), (100, 5), (220, 3)],
              # pairs of training depth and number of training iterations for the depth
              initial_state=None,
              lambd=1,
              alpha_conn=1.,
              alpha_diss=1.,
              alpha_energy=1.,
              alpha_heat=1.,
              alpha_max_min=1.,
              t_max=1e3,
              lr=1e-5,
              betas=(.5, .999),
              save_best=False,
              save_path='Train_TRAGD.pt',
              obj='pyramid_9',
              min_layer=571,
              max_layer=958,
              vertex_multipliers=None,
              set_plot_limits=True,
              whole_layer=False,
              device=device):
    n_layers = depths_iters[-1][0]  # total number of layers in the final part

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    loss_fn = nn.MSELoss()

    full_iters = depths_iters[-1][1]
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1., end_factor=1. / full_iters,
                                                  total_iters=full_iters)

    history = []
    energy_loss = []
    start_time = int(time.time())

    save_depth = False

    # increase the number of layers used for training, as later layers are sensitive to the hidden state obtained from the model predictions, and
    # thus require a relatively well-trained model in order to produce reasonable predictions
    for dix, (depth, depth_iterations) in enumerate(depths_iters):
        print('depth:', depth)

        if dix == (len(depths_iters) - 1):
            save_depth = True
            best_loss = np.inf

        if not whole_layer and depth >= 5:
            ditr = range(depth_iterations)
            pbar = True
        else:
            ditr = tqdm(range(depth_iterations))
            pbar = False

        for depth_iter in ditr:

            # initialize values for accumulating the losses for the current training depth
            depth_avg_loss = 0.
            depth_fwd_loss = 0.

            if pbar:
                itr = tqdm(range(depth))
            else:
                itr = range(depth)

            for layer in itr:

                data = load_data_layer(layer, obj=obj, layers=f'{min_layer}_to_{max_layer}', min_layer=min_layer,
                                       max_layer=max_layer, vertex_multipliers=vertex_multipliers, device=device)

                # initialize values for accumulating losses for the current layer
                layer_loss = 0.
                layer_const_loss = 0.
                layer_conn_loss = 0.
                layer_diss_loss = 0.
                layer_energy = 0.
                layer_avg_steps = 0.

                if layer == 0:
                    if initial_state is not None:
                        layer_init_state = initial_state
                    else:
                        layer_init_state = model.boundary_value * torch.ones((data[0].shape[0],), dtype=torch.float32,
                                                                             device=device)

                # determine the initial heat state for the layer by transfering the final state of the previous layer, taking into account the
                # diffusion in the time passed during recoating
                else:
                    time_delta = (data[4][0] - prev_last_time) / 1000
                    state = model(prev_distances, prev_densities, prev_boundary, state, time_delta).detach().view(-1)
                    layer_init_state = transfer_state(prev_vertices, state, data[0])

                prev_vertices = data[0]
                prev_distances = data[1]
                prev_densities = data[2]
                prev_boundary = data[3]
                prev_last_time = data[4][-1]

                state = layer_init_state

                # sample the evaluation time-steps (either random or n-step)

                # jumps = np.random.poisson(lambd,len(data[5])) + 1
                # eval_ixs = np.cumsum(jumps)
                # eval_ixs = eval_ixs[np.where(eval_ixs<len(data[5]))].tolist()
                # if eval_ixs[0] > 0:
                #     eval_ixs = [0,] + eval_ixs

                if whole_layer:
                    eval_ixs = [0, len(data[5]) - 1]
                else:
                    eval_ixs = np.arange(0, len(data[5]), lambd).tolist()
                    if eval_ixs[-1] < len(data[5]) - 1:
                        eval_ixs.append(len(data[5]) - 1)

                X = [(data[1], data[2], data[3], data[5][eval_ixs[i]],
                      (data[4][eval_ixs[i + 1]] - data[4][eval_ixs[i]]) / 1000, data[6][eval_ixs[i + 1]]) for i in
                     range(len(eval_ixs) - 1)]
                Y = [data[5][eval_ixs[i]] for i in range(1, len(eval_ixs))]

                for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X, Y):
                    optim.zero_grad()

                    # assign the surface temperature values
                    state[surface_temp.indices()[0]] = 0.
                    state = state + surface_temp.to_dense()

                    # evaluate the model and loss function
                    pred, conn_loss, diss_loss, diss_vec, laser_heat, edge_ind, edge_attr = model(distances.to(device), densities.to(device),
                                                                             boundary.to(device), state.to(device),
                                                                             time_delta.to(device),
                                                                             laser_dist.to(device), fit=True)
                    pred_loss = loss_fn(pred[y.indices()].view(-1, 1), y.values().view(-1, 1))

                    # calculate the regularizing loss functions
                    pred = pred - laser_heat
                    neighbor_temp = (torch.sign(distances).to_dense() * pred.view(-1).to_dense())
                    max_principle_violation = (torch.relu(pred.view(-1) - neighbor_temp.max(dim=1).values)).where(
                        torch.logical_and(pred > state, boundary.sum(dim=1) == 0),
                        torch.zeros_like(pred)).square().mean()
                    min_principle_violation = (torch.relu(neighbor_temp.min(dim=1).values - pred.view(-1))).where(
                        torch.logical_and(pred > state, boundary.sum(dim=1) == 0),
                        torch.zeros_like(pred)).square().mean()
                    energy_violation = torch.relu(
                        calculate_energy(pred, densities, boundary)  # energy of prediction
                        - calculate_energy(state, densities, boundary)  # energy of previous state
                    )
                    heat, heat_diss = calculate_heat(pred, densities, boundary, dissipation=diss_vec.squeeze())
                    total_heat_diff = ((heat - heat_diss) / state.sum() - 1).square()

                    # assemble the total loss
                    loss = pred_loss + \
                           alpha_conn * conn_loss + \
                           alpha_diss * diss_loss + \
                           alpha_energy * energy_violation + \
                           alpha_heat * total_heat_diff + \
                           alpha_max_min * (max_principle_violation + min_principle_violation)

                    # optimization step
                    loss.backward()
                    optim.step()

                    # log the loss values
                    push_fwd_loss = loss_fn(state[y.indices()].view(-1, 1), y.values().view(-1, 1))
                    layer_loss += pred_loss.item() / (len(eval_ixs) - 1)
                    layer_const_loss += push_fwd_loss.item() / (len(eval_ixs) - 1)
                    layer_conn_loss += conn_loss.item() / (len(eval_ixs) - 1)
                    layer_diss_loss += diss_loss.item() / (len(eval_ixs) - 1)
                    layer_energy += (
                                                energy_violation + total_heat_diff + max_principle_violation + min_principle_violation).item() / (
                                                len(eval_ixs) - 1)

                    # use the predicted state for the next time-step
                    state = pred.detach().clip(0, t_max).view(-1)

                layer_avg_steps += len(eval_ixs) - 1

                if pbar:
                    if layer_const_loss != 0.:
                        itr.set_postfix({'l_T': layer_loss,
                                         'rel': (layer_const_loss - layer_loss) / layer_const_loss,
                                         'l_reg': (layer_conn_loss + layer_diss_loss),
                                         'l_E': layer_energy,
                                         't_min': state.min().item(),
                                         't_max': state.max().item()})

                # energy_loss.append(layer_energy)
                depth_avg_loss += layer_loss
                depth_fwd_loss += layer_const_loss
            if not depth_fwd_loss == 0:
                if pbar:
                    print(depth_avg_loss / depth, (depth_fwd_loss - depth_avg_loss) / depth_fwd_loss)
                else:
                    ditr.set_postfix(
                        {'l_T': depth_avg_loss / depth, 'rel': (depth_fwd_loss - depth_avg_loss) / depth_fwd_loss})
            if save_depth:
                scheduler.step()
                if depth_avg_loss < best_loss:
                    best_loss = depth_avg_loss
                    print('saving model state.')
                    model.save(save_path, compiled=True, override=True)

        # plot_state(data[0].detach().numpy(), state.detach().numpy(),data[7], set_limits=set_plot_limits, save_path=f'plots/depth_{depth}_plot{dix}.svg')
        print('parameters:')
        print(model.diss_model.coefs)
        print(model.laser_model.intensity.item(), ', ', model.laser_model.decay.item(), sep='')

    return history

############################################ fit_multi_model ###############################################

def develop_layers(model,
                   n_layers=500,
                   print_layers=[5, 6, 7, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 232, 250, 300, 350, 380, 450, 500],
                   boundary_value=124.9,
                   use_data=None,
                   set_limits=True,
                   obj='pyramid_7',
                   vertex_multipliers=None,
                   plot_dir='./plots/layer_plots'):
    """
    evaluate the model and visualize the predicted heat states
    """

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    state = None
    history = []
    energy_loss = []
    maximum_temp = []
    itr = tqdm(range(n_layers))

    for layer in itr:
        if obj == 'pyramid_7':
            data = load_data_layer(layer)
        else:
            data = load_data_layer(layer, obj=obj, layers="90_to_327", min_layer=90, max_layer=327, vertex_multipliers = vertex_multipliers)

        if layer == 0:
            layer_init_state = boundary_value * torch.ones((data[0].shape[0],), dtype=torch.float32,device=device)
        else:
            time_delta = (data[4][0]-prev_last_time)/1000
            #state = model(prev_distances, prev_densities, prev_boundary, state, time_delta).detach().view(-1)
            ###############################
            state = model(prev_distances, prev_densities, prev_boundary, state, time_delta).detach().view(-1)
            ######################################
            layer_init_state = transfer_state(prev_vertices, state, data[0])
        prev_vertices = data[0]
        prev_distances = data[1]
        prev_densities = data[2]
        prev_boundary = data[3]
        prev_last_time = data[4][-1]

        state = layer_init_state
        eval_ixs = np.arange(0,len(data[5]),1).tolist()

        if eval_ixs[-1] < len(data[5])-1:
            eval_ixs.append(len(data[5]) - 1)

        X = [(data[1], data[2], data[3], data[5][eval_ixs[i]], (data[4][eval_ixs[i+1]]-data[4][eval_ixs[i]])/1000, data[6][eval_ixs[i+1]]) for i in range(len(eval_ixs)-1)]
        Y = [data[5][eval_ixs[i]] for i in range(1, len(eval_ixs))]

        X_1 = []
        Y_1 = []
        X_1.append(X[0])
        X_1.append(X[-1])
        Y_1.append(Y[0])
        Y_1.append(Y[-1])

        layer_pred_quality = 0.
        layer_cooling_rate = 0.
        layer_max_temp = 0.
        layer_energy = 0.
        counter = 0
        pred = 0.0
        max_temp =[]
        timestep = 0
        for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X,Y):
            if use_data is None or layer in use_data:
                state[surface_temp.indices()[0]] = 0.
                state = state + surface_temp.to_dense()
            pred_previous = pred
            #pred = model(distances, densities, boundary, state, time_delta, laser_dist)

            pred, conn_loss, diss_loss, diss_vec, laser_heat = model(distances.to(device), densities.to(device),
                                                                     boundary.to(device), state.to(device),
                                                                     time_delta.to(device), laser_dist.to(device),
                                                                     fit=True)
            ###########################################
            neighbor_temp = (torch.sign(distances).to_dense() * pred.view(-1).to_dense())
            max_principle_violation = (torch.relu(pred.view(-1) - neighbor_temp.max(dim=1).values)).where(
                torch.logical_and(pred > state, boundary.sum(dim=1) == 0), torch.zeros_like(pred)).square().mean()
            min_principle_violation = (torch.relu(neighbor_temp.min(dim=1).values - pred.view(-1))).where(
                torch.logical_and(pred > state, boundary.sum(dim=1) == 0), torch.zeros_like(pred)).square().mean()
            energy_violation = torch.relu(
                calculate_energy(pred, densities, boundary)  # energy of prediction
                - calculate_energy(state, densities, boundary)  # energy of previous state
            )
            #print(energy_violation.item())
            heat, heat_diss = calculate_heat(pred, densities, boundary, dissipation=diss_vec.squeeze())
            total_heat_diff = ((heat - heat_diss) / state.sum() - 1).square()
            layer_energy += (energy_violation + total_heat_diff + max_principle_violation + min_principle_violation).item() / (len(data[5])-1)


            ##########################################

            pred_true = torch.stack([((state - pred)[surface_temp.indices()[0]]),
                                        ((state-y)[surface_temp.indices()[0]])])


            pred_quality = torch.sqrt(torch.mean(torch.square(pred[y.indices()].view(-1,1) - y.values().view(-1, 1))))/ torch.mean(torch.square(y.values().view(-1, 1)-torch.mean(y.values().view(-1, 1))))
            pred_quality = torch.nan_to_num(pred_quality).item()

            pred_quality = torch.sqrt(torch.mean(torch.square(pred[y.indices()].view(-1,1) - y.values().view(-1, 1))))/ torch.mean(torch.square(y.values().view(-1, 1)-torch.mean(y.values().view(-1, 1))))
            max_tempperature  = torch.max(pred).item()
            max_temp.append(max_tempperature)
            pred_quality = torch.nan_to_num(pred_quality).item()
            layer_pred_quality += pred_quality/(len(data[5])-1)

            state = pred.detach().clip(0, 1e3).view(-1)
            counter = counter + 1

        maximum_temp.append(max_temp)
        history.append(layer_pred_quality)
        energy_loss.append(layer_energy) # weg
        itr.set_postfix({'corr': layer_pred_quality})
        counter = 0
        if (layer+1 in print_layers): # in print_layers
            #bounds = {'x': (0, 30), 'y': (0, 23), 'z': (0, 12)}
          plot_state(data[0].detach().numpy(), state.detach().numpy(), data[7], set_limits=set_limits, C_to_K=set_limits, show=False, save_path=f'{plot_dir}/layer_{layer+1}_same_range.pdf', plot_format='pdf')
          layer_temp = load_surface_temperatures(layer, obj=obj, start=(571 if obj=='pyramid_7' else 0))[-1]
          plot_surface(layer_temp, save_path=f'{plot_dir}/surface_data_layer_{layer+1}.pdf', set_limits=set_limits,show=False, C_to_K=set_limits)
          plot_surface_state(data[0].detach().numpy(), state.detach().numpy(), save_path=f'{plot_dir}/surface_layer_{layer+1}.pdf', set_limits=set_limits, C_to_K=set_limits, show=False, vertex_multipliers=vertex_multipliers)


    return history


def develop_layers_state(model, n_layers=500, boundary_value=124.9, use_data=None, obj=None, layers=None, return_all=False):

    if obj is None:
        obj = "pyramid_7"
    if layers is None:
        if obj == "pyramid_7":
            layers = "571_to_1079"
        elif obj == "pyramid_3":
            layers = "571_to_1064"
    
    state = None
    itr = tqdm(range(n_layers))
    
    data_losses = []
    consistency_losses = []
    heateq_losses = []
    layer_correlations = []
    
    loss_fn = nn.MSELoss()
    
    ts = int(dt.datetime.timestamp(dt.datetime.now()))
    dirname = f'eval_plots_{ts}'
    # os.mkdir(dirname)

    if return_all:
        states = []

    for layer in itr:
        if obj is None:
            data = load_data_layer(layer)
        else:
            data = load_data_layer(layer,obj=obj, layers=layers)
        
        layer_data_loss = 0.
        layer_consistency_loss = 0.
        layer_heateq_loss = 0.
    
        
        if layer==0:
            layer_init_state = boundary_value * torch.ones((data[0].shape[0],),dtype=torch.float32,device=device)
        else:
            time_delta=(data[4][0]-prev_last_time)/1000
            state = model(prev_distances, prev_densities, prev_boundary, state, time_delta).detach().view(-1)
            layer_init_state = transfer_state(prev_vertices,state,data[0])
        prev_vertices = data[0]
        prev_distances = data[1]
        prev_densities = data[2]
        prev_boundary = data[3]
        prev_last_time = data[4][-1]
        
        state = layer_init_state
        
        eval_ixs = np.arange(0,len(data[5]),1).tolist()
        if eval_ixs[-1] < len(data[5])-1:
            eval_ixs.append(len(data[5])-1)

        X = [(data[1], data[2], data[3], data[5][eval_ixs[i]], (data[4][eval_ixs[i+1]]-data[4][eval_ixs[i]])/1000, data[6][eval_ixs[i+1]]) for i in range(len(eval_ixs)-1)]
        Y = [data[5][eval_ixs[i]] for i in range(1,len(eval_ixs))]

        layer_pred_quality = 0.
        for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X,Y):
            if use_data is None or use_data[layer]:
                state[surface_temp.indices()[0]] = 0.
                state = state + surface_temp.to_dense()
            
            # pred = model(distances, densities, boundary, state, time_delta, laser_dist)
            pred = model(distances, densities, boundary, state, time_delta, laser_dist)
            state = pred.detach().clip(0,1e3).view(-1)
        if return_all:
            states.append(state.clone())
    if return_all:
        return states
    return state

def predict_layer(model,state,layer, prev_distances, prev_vertices, prev_densities, prev_boundary, prev_last_time, loss_fn = nn.MSELoss(), boundary_value=124.9, use_data=None, obj=None, layers=None, plot_states=False):
    if obj is None:
        data = load_data_layer(layer)
    else:
        data = load_data_layer(layer,obj=obj, layers=layers)
    
    layer_data_loss = 0.
    layer_consistency_loss = 0.
    layer_heateq_loss = 0.
    ######################
    energy_violation_loss = 0.
    conn_violation_loss = 0.
    diss_violation_loss = 0.
    ######################

    
    if layer==0:
        layer_init_state = boundary_value * torch.ones((data[0].shape[0],),dtype=torch.float32,device=device)
    else:
        time_delta=(data[4][0]-prev_last_time)/1000
        state = model(prev_distances, prev_densities, prev_boundary, state, time_delta).detach().view(-1)
        layer_init_state = transfer_state(prev_vertices,state,data[0])
    prev_vertices = data[0]
    prev_distances = data[1]
    prev_densities = data[2]
    prev_boundary = data[3]
    prev_last_time = data[4][-1]
    
    state = layer_init_state
    
    eval_ixs = np.arange(0,len(data[5]),1).tolist()
    if eval_ixs[-1] < len(data[5])-1:
        eval_ixs.append(len(data[5])-1)

    X = [(data[1], data[2], data[3], data[5][eval_ixs[i]], (data[4][eval_ixs[i+1]]-data[4][eval_ixs[i]])/1000, data[6][eval_ixs[i+1]]) for i in range(len(eval_ixs)-1)]
    Y = [data[5][eval_ixs[i]] for i in range(1,len(eval_ixs))]

    layer_pred_quality = 0.
    for ix, ((distances, densities, boundary, surface_temp, time_delta, laser_dist), y) in enumerate(zip(X,Y)):
        if use_data is None or layer in use_data:
            state[surface_temp.indices()[0]] = 0.
            state = state + surface_temp.to_dense()
        
        # pred = model(distances, densities, boundary, state, time_delta, laser_dist)
        pred, conn_loss, diss_loss, diss_vec, laser_heat = model(distances, densities, boundary, state, time_delta, laser_dist, fit=True)

        layer_data_loss += loss_fn(pred[y.indices()].view(-1,1),y.values().view(-1,1)).item()/(len(data[5])-1)

        pred = pred - laser_heat
        neighbor_temp = (torch.sign(distances).to_dense() * pred.view(-1).to_dense())
        max_principle_violation = (torch.relu(pred.view(-1)-neighbor_temp.max(dim=1).values)).where(torch.logical_and(pred > state,boundary.sum(dim=1)==0),torch.zeros_like(pred)).square().mean()
        min_principle_violation = (torch.relu(neighbor_temp.min(dim=1).values-pred.view(-1))).where(torch.logical_and(pred > state,boundary.sum(dim=1)==0),torch.zeros_like(pred)).square().mean()
        energy_violation = torch.relu(
                    calculate_energy(pred,densities,boundary) #energy of prediction
                    - calculate_energy(state,densities,boundary) #energy of previous state
                    )
        heat, heat_diss = calculate_heat(pred,densities,boundary, dissipation=diss_vec.squeeze())
        total_heat_diff = ((heat-heat_diss)/state.sum()-1).square()

        layer_consistency_loss += (conn_loss + diss_loss + total_heat_diff).item()/(len(data[5])-1)
        layer_heateq_loss += (energy_violation + max_principle_violation + min_principle_violation).item()/(len(data[5])-1)
        ####################################################################
        energy_violation_loss += (energy_violation).item()/(len(data[5])-1)
        conn_violation_loss += (conn_loss).item()/(len(data[5])-1)
        diss_violation_loss +=(diss_loss).item()/(len(data[5])-1)
        ####################################################################

        pred_true = torch.stack([((state-pred)[surface_temp.indices()[0]]),
                                    ((state-y)[surface_temp.indices()[0]])])
        pred_quality = torch.corrcoef(pred_true)[1,0]
        pred_quality = torch.nan_to_num(pred_quality).item()
        layer_pred_quality += pred_quality/(len(data[5])-1)

        state = pred.detach().clip(0,1e3).view(-1)
        if plot_states:
            plot_state(prev_vertices,state,np.array(data[7]),save_path=f'plots/animation_plots/{str(layer).zfill(4)}_{str(ix).zfill(3)}.png', show=False,plot_format='png',backend='agg')
    return state, (prev_vertices, prev_distances, prev_densities, prev_boundary, prev_last_time), (layer_data_loss,layer_consistency_loss, layer_heateq_loss, layer_pred_quality, conn_violation_loss, diss_violation_loss, energy_violation_loss)

        
def develop_layers_compare(model1,model2,n_layers=500,print_layers=[5,10,25,50,100,150,200,250,300,350,400,450,500],boundary_value=124.9, use_data=None, obj=None, layers=None):
    """
    evaluate the model and visualize the predicted heat states
    """
    
    if obj is None:
        obj = "pyramid_7"
    if layers is None:
        layers = "571_to_1079"
    
    state1, state2 = None, None
    itr = tqdm(range(n_layers))
    
    data_losses = []
    consistency_losses = []
    heateq_losses = []
    layer_correlations = []
    
    loss_fn = nn.MSELoss()
    
    ts = int(dt.datetime.timestamp(dt.datetime.now()))
    dirname = f'eval_plots_{ts}'
    os.mkdir(dirname)

    prev_vertices1 = None
    prev_distances1 = None
    prev_densities1 = None
    prev_boundary1 = None
    prev_last_time1 = None

    prev_vertices2 = None
    prev_distances2 = None
    prev_densities2 = None
    prev_boundary2 = None
    prev_last_time2 = None

    for layer in itr:
        state1, \
            (prev_vertices1, prev_distances1, prev_densities1, prev_boundary1, prev_last_time1), \
            (layer_data_loss1,layer_consistency_loss1, layer_heateq_loss1, layer_pred_quality1) \
                    = predict_layer(model1,state1,layer, prev_distances1, prev_vertices1, prev_densities1, prev_boundary1, prev_last_time1, loss_fn=loss_fn, boundary_value=boundary_value, use_data=use_data, obj=obj, layers=layers)

        state2, \
            (prev_vertices2, prev_distances2, prev_densities2, prev_boundary2, prev_last_time2), \
            (layer_data_loss2,layer_consistency_loss2, layer_heateq_loss2, layer_pred_quality2) \
                    = predict_layer(model2,state2,layer, prev_distances2, prev_vertices2, prev_densities2, prev_boundary2, prev_last_time2, loss_fn=loss_fn, boundary_value=boundary_value, use_data=use_data, obj=obj, layers=layers)

        data_losses.append((layer_data_loss1,layer_data_loss2))
        consistency_losses.append((layer_consistency_loss1,layer_consistency_loss2))
        heateq_losses.append((layer_heateq_loss1,layer_heateq_loss2))
        layer_correlations.append((layer_pred_quality1,layer_pred_quality2))
        itr.set_postfix({'l_data':(layer_data_loss1,layer_data_loss2),'l_consistency':(layer_consistency_loss1,layer_consistency_loss2), 'l_heateq':(layer_heateq_loss1,layer_heateq_loss2),'corr':(layer_pred_quality1,layer_pred_quality2)})
        if (layer+1) in print_layers:
            data = load_data_layer(layer)
            plot_diff(data[0].detach().numpy(), (state1-state2).detach().numpy(),data[7], save_path=f'{dirname}/layer_{layer+1}_compare.pdf', show= False)
            layer_temp = load_surface_temperatures(layer)[-1]
            plot_surface_diff(data[0].detach().numpy(), (state1-state2).detach().numpy(), save_path=f'{dirname}/surface_layer_{layer+1}_compare.pdf', show= False)

    return (state1, state2), data_losses, consistency_losses, heateq_losses, layer_correlations
        

def develop_layers_eval(model, n_layers=500, print_layers=[5,10,25,50,100,150,200,250,300,350,400,450,500],boundary_value=124.9, use_data=None, obj=None, layers=None, all_plots=False):
    """
    evaluate the model and visualize the predicted heat states
    """
    
    if obj is None:
        obj = "pyramid_7"
    if layers is None:
        layers = "571_to_1079"
    
    state = None
    itr = tqdm(range(n_layers))
    
    data_losses = []
    consistency_losses = []
    heateq_losses = []
    layer_correlations = []
    
    loss_fn = nn.MSELoss()
    
    ts = int(dt.datetime.timestamp(dt.datetime.now()))
    dirname = f'eval_plots_{ts}'
    # os.mkdir(dirname)

    prev_vertices = None
    prev_distances = None
    prev_densities = None
    prev_boundary = None
    prev_last_time = None

    for layer in itr:
        state, \
            (prev_vertices, prev_distances, prev_densities, prev_boundary, prev_last_time), \
            (layer_data_loss,layer_consistency_loss, layer_heateq_loss, layer_pred_quality) \
                    = predict_layer(model,state,layer, prev_distances, prev_vertices, prev_densities, prev_boundary, prev_last_time, loss_fn=loss_fn, boundary_value=boundary_value, use_data=use_data, obj=obj, layers=layers, plot_states=all_plots)
        # if obj is None:
        #     data = load_data_layer(layer)
        # else:
        #     data = load_data_layer(layer,obj=obj, layers=layers)
        
        # layer_data_loss = 0.
        # layer_consistency_loss = 0.
        # layer_heateq_loss = 0.
    
        
        # if layer==0:
        #     layer_init_state = boundary_value * torch.ones((data[0].shape[0],),dtype=torch.float32,device=device)
        # else:
        #     time_delta=(data[4][0]-prev_last_time)/1000
        #     state = model(prev_distances, prev_densities, prev_boundary, state, time_delta).detach().view(-1)
        #     layer_init_state = transfer_state(prev_vertices,state,data[0])
        # prev_vertices = data[0]
        # prev_distances = data[1]
        # prev_densities = data[2]
        # prev_boundary = data[3]
        # prev_last_time = data[4][-1]
        
        # state = layer_init_state
        
        # eval_ixs = np.arange(0,len(data[5]),1).tolist()
        # if eval_ixs[-1] < len(data[5])-1:
        #     eval_ixs.append(len(data[5])-1)

        # X = [(data[1], data[2], data[3], data[5][eval_ixs[i]], (data[4][eval_ixs[i+1]]-data[4][eval_ixs[i]])/1000, data[6][eval_ixs[i+1]]) for i in range(len(eval_ixs)-1)]
        # Y = [data[5][eval_ixs[i]] for i in range(1,len(eval_ixs))]

        # layer_pred_quality = 0.
        # for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X,Y):
        #     if use_data is None or use_data[layer]:
        #         state[surface_temp.indices()[0]] = 0.
        #         state = state + surface_temp.to_dense()
            
        #     # pred = model(distances, densities, boundary, state, time_delta, laser_dist)
        #     pred, conn_loss, diss_loss, diss_vec, laser_heat = model(distances, densities, boundary, state, time_delta, laser_dist, fit=True)

        #     layer_data_loss += loss_fn(pred[y.indices()].view(-1,1),y.values().view(-1,1)).item()/(len(data[5])-1)

        #     pred = pred - laser_heat
        #     neighbor_temp = (torch.sign(distances).to_dense() * pred.view(-1).to_dense())
        #     max_principle_violation = (torch.relu(pred.view(-1)-neighbor_temp.max(dim=1).values)).where(torch.logical_and(pred>state,boundary.sum(dim=1)==0),torch.zeros_like(pred)).square().mean()
        #     min_principle_violation = (torch.relu(neighbor_temp.min(dim=1).values-pred.view(-1))).where(torch.logical_and(pred>state,boundary.sum(dim=1)==0),torch.zeros_like(pred)).square().mean()
        #     energy_violation = torch.relu(
        #                 calculate_energy(pred,densities,boundary) # energy of prediction 
        #                -calculate_energy(state,densities,boundary) # energy of previous state
        #              )
        #     heat, heat_diss = calculate_heat(pred,densities,boundary, dissipation=diss_vec.squeeze())
        #     total_heat_diff = ((heat-heat_diss)/state.sum()-1).square()

        #     layer_consistency_loss += (conn_loss + diss_loss + total_heat_diff).item()/(len(data[5])-1)
        #     layer_heateq_loss += (energy_violation + max_principle_violation + min_principle_violation).item()/(len(data[5])-1)


        #     pred_true = torch.stack([((state-pred)[surface_temp.indices()[0]]),
        #                                 ((state-y)[surface_temp.indices()[0]])])
        #     pred_quality = torch.corrcoef(pred_true)[1,0]
        #     pred_quality = torch.nan_to_num(pred_quality).item()
        #     layer_pred_quality += pred_quality/(len(data[5])-1)

        #     state = pred.detach().clip(0,1e3).view(-1)

        
        data_losses.append(layer_data_loss)
        consistency_losses.append(layer_consistency_loss)
        heateq_losses.append(layer_heateq_loss)
        layer_correlations.append(layer_pred_quality)
        itr.set_postfix({'l_data':layer_data_loss,'l_consistency':layer_consistency_loss, 'l_heateq':layer_heateq_loss,'corr':layer_pred_quality})
        # if (layer+1) in print_layers:
        # if layer>0:
        #     plot_state(data[0].detach().numpy(), state.detach().numpy(),data[7], save_path=f'{dirname}/layer_{layer+1}_same_range.svg',show=False)
        #     layer_temp = load_surface_temperatures(layer)[-1]
        #     plot_surface(layer_temp, save_path=f'{dirname}/surface_data_layer_{layer+1}.svg',show=False)
        #     plot_surface_state(data[0].detach().numpy(), state.detach().numpy(), save_path=f'{dirname}/surface_layer_{layer+1}.eps',show=False)

    return data_losses, consistency_losses, heateq_losses, layer_correlations


def develop_layers_eval_full_layer(model, n_layers=500, print_layers=[5,10,25,50,100,150,200,250,300,350,400,450,500],boundary_value=124.9, obj=None, layers=None, all_plots=False):
    """
    evaluate the model and visualize the predicted heat states
    """
    
    if obj is None:
        obj = "pyramid_7"
    if layers is None:
        layers = "571_to_1079"
    
    state = None
    itr = tqdm(range(n_layers))
    
    data_losses = []
    consistency_losses = []
    heateq_losses = []
    layer_correlations = []
    conn_losses = []
    diss_losses = []
    energy_losses = []
    
    loss_fn = nn.MSELoss()
    
    ts = int(dt.datetime.timestamp(dt.datetime.now()))
    dirname = f'eval_plots_{ts}'
    # os.mkdir(dirname)

    prev_vertices = None
    prev_distances = None
    prev_densities = None
    prev_boundary = None
    prev_last_time = None

    predictions = []

    for layer in itr:
        layer_data = load_data_layer(layer)
        init_state = layer_data[5][0]
        state, \
            (prev_vertices, prev_distances, prev_densities, prev_boundary, prev_last_time), \
            (layer_data_loss,layer_consistency_loss, layer_heateq_loss, layer_pred_quality, layer_conn_loss, layer_diss_loss, layer_energy_loss) \
                    = predict_layer(model,state,layer, prev_distances, prev_vertices, prev_densities, prev_boundary, prev_last_time, loss_fn=loss_fn, boundary_value=boundary_value, use_data=[], obj=obj, layers=layers,plot_states=all_plots)

        final_state = layer_data[5][-1]
        
        # print(init_state)
        # print()
        # print(final_state)
        
        predicted_change = (state[init_state.indices()[0]] - init_state.values())
        true_change = (final_state.values() - init_state.values())

        predictions.append((init_state.values(),true_change,predicted_change,final_state.values()))
        consistency_losses.append(layer_consistency_loss)
        heateq_losses.append(layer_heateq_loss)
        conn_losses.append(layer_conn_loss)
        diss_losses.append(layer_diss_loss)
        energy_losses.append(layer_energy_loss)

    all_predictions = torch.cat([t[1] for t in predictions], dim=0)
    all_actual_changes = torch.cat([t[2] for t in predictions], dim=0)

    layer_losses_normal = [np.linalg.norm(t[1]-t[2])/np.linalg.norm(t[0]) for t in predictions]
    consistency_losses_normal = [np.sqrt(loss)/np.linalg.norm(t[0]) for t,loss in zip(predictions,consistency_losses)]
    heateq_losses_normal = [np.sqrt(loss)/np.linalg.norm(t[0]) for t, loss in zip(predictions,heateq_losses)]
    conn_loss_normal = [np.sqrt(loss)/np.linalg.norm(t[0]) for t, loss in zip(predictions, conn_losses)]
    diss_loss_normal = [np.sqrt(loss)/np.linalg.norm(t[0]) for t, loss in zip(predictions, diss_losses)]
    energy_loss_normal = [np.sqrt(loss)/np.linalg.norm(t[0]) for t, loss in zip(predictions, energy_losses)]

    predictions_np = all_predictions.numpy()
    actual_np = all_actual_changes.numpy()
    dsnp = np.stack([predictions_np,actual_np])
    print(np.corrcoef(dsnp, rowvar=True)[0,1])

    return predictions, layer_losses_normal, consistency_losses_normal, heateq_losses_normal, conn_loss_normal, diss_loss_normal, energy_loss_normal
    