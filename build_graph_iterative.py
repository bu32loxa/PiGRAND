import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
import alphashape

import pickle as pkl
import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from copy import copy, deepcopy
from traceback import print_exception


BOUNDARY_TEMP = 150


def identify_vertices(layer_images):
    """
    identify pixels that belong to the part based on the temperature
    """

    vertices = np.zeros_like(layer_images[0])
    for image in layer_images:
        max_temp = image.max()
        vertices += np.where(image > max(BOUNDARY_TEMP + .30 * (max_temp - BOUNDARY_TEMP), BOUNDARY_TEMP + 100), 1, 0)
        # vertices += np.where(image>300,1,0)
    vertices = vertices / len(layer_images)
    return vertices


def grid_vertices_to_vectors(vertices, z=None):
    """
    takes a grid of boolean values and returns the indices of 1s
    """

    indices = np.indices(vertices.shape).transpose(1, 2, 0)
    ret_val = indices[np.where(vertices)]
    if z is not None:
        ret_val = np.concatenate([ret_val, z * np.ones((len(ret_val), 1))], axis=-1)
    return ret_val


inv_ = np.vectorize(lambda x: 1 / x if not x == 0. else 0.)


def scale_invariant_density(space, return_avg_dist=False):
    """
    calculate the scale-invariant-density, as defined in: https://doi.org/10.48550/arXiv.2110.01286
    with adaptation for 3d data
    """

    ret_val = None
    dim = space.shape[-1]
    if dim != 2 and dim != 3:
        print(space.shape)
        raise NotImplementedError()
    pairings = np.tile(space, (space.shape[0], 1, 1)) - np.tile(space, (space.shape[0], 1, 1)).transpose([1, 0, 2])
    dens = np.sum(np.square(pairings), axis=-1)
    if dim == 2:
        ret_val = np.sum(inv_(np.sqrt(dens)), axis=1)
    else:
        ret_val = np.sum(inv_(dens), axis=1)

    if return_avg_dist:
        return ret_val, np.linalg.norm(pairings.reshape(-1, dim), axis=-1).mean()
    return ret_val


def prune_space(space, n=30):
    """
    takes a point cloud ('space') and prunes it to the specified number of points, using the sid-method
    """

    pruned_space = space.copy()
    for _ in range(len(space) - n):  # iteratively remove the point with the highest sid-value
        densities = scale_invariant_density(pruned_space)
        prune_ix = np.argmax(densities)
        pruned_space = np.concatenate([pruned_space[:prune_ix], pruned_space[prune_ix + 1:]])
    return pruned_space


# retrieve the layer files created using 'LayerSeperation_2.py'

object_name = 'pyramid_8'
layers = os.listdir(object_name)

for i, f in enumerate(layers):
    if 'file_layer_map' in f:
        layers.pop(i)
        break

layers = [(int(l.split('_')[1].split('.')[0]), l) for l in layers]
layers.sort(key=lambda x: x[0])

inv_ = np.vectorize(lambda x: 1 / x if not x == 0. else 0.)


def unmask_ix(ix, mask):
    """
    from a boolean mask, get the ix-th 1-index
    """

    ixs = (np.cumsum(mask) - 1)
    full_ix = np.searchsorted(ixs, ix)
    return full_ix


def estimate_density(target_dist, d=3):
    """
    estimate the sid for points in the full lattice
    """

    ran = target_dist * np.linspace(-20, 20, 41)
    lattice = np.stack(np.meshgrid(*[ran for _ in range(d)])).reshape(-1, d)
    sq_dist = np.apply_along_axis(lambda x: x.T @ x, -1, lattice)
    density = np.sum(inv_(sq_dist))
    return .5 * density


def prune_selective(vertices, prunable_mask, density_mask, target_dist):
    """
    prune a point cloud restricting a) which points can be pruned and b) which points should be taken into account
    when calculating the sid
    """

    target_density = estimate_density(target_dist)
    for _ in range(len(vertices[prunable_mask])):
        prunable_vertices = vertices[prunable_mask]
        density_vertices = vertices[density_mask]
        pairings = np.tile(prunable_vertices, (density_vertices.shape[0], 1, 1)).transpose(1, 0, 2) - np.tile(
            density_vertices, (prunable_vertices.shape[0], 1, 1))
        dens = np.sum(np.square(pairings), axis=-1)
        point_density = np.sum(inv_(dens), axis=1)
        prune_ix = np.argmax(point_density)
        if np.mean(point_density) <= target_density:
            break
        full_prune_ix = unmask_ix(prune_ix, prunable_mask)
        vertices = np.delete(vertices, full_prune_ix, axis=0)
        prunable_mask = np.delete(prunable_mask, full_prune_ix, axis=0)
        density_mask = np.delete(density_mask, full_prune_ix, axis=0)
    return vertices


def interior_points_random(simplex_coords, n=100):
    """
    | sample random points in simplex, distribution according to:
    | https://mathoverflow.net/a/368231
    """

    dim = simplex_coords.shape[0]
    weights = np.random.exponential(1, (n, dim))
    weights = (weights.T / weights.sum(axis=1)).T
    return weights @ simplex_coords


def get_good_simplices(simplices, space, alpha_shape, thresh=.5):
    """
    | identify the simplices that are predominantly located inside the alphashape
    """

    good_simplices = []
    for simplex in tqdm(simplices):
        simplex_coords = np.array([space[v] for v in simplex])
        points = interior_points_random(simplex_coords)
        count = alpha_shape.contains(points).sum()
        if count / len(points) > thresh:
            good_simplices.append(simplex)
    return np.array(good_simplices)


def check_simplex(simplex, space, alpha_shape, thresh=.5):
    """
    | check for a single simplex, whether it is predominantly inside the alphashape
    """

    simplex_coords = np.array([space[v] for v in simplex])
    points = interior_points_random(simplex_coords)
    count = alpha_shape.contains(points).sum()
    if count / len(points) > thresh:
        return True
    else:
        return False


def get_good_simplices_parallel(simplices, space, alpha_shape, thresh=.5):
    mask = Parallel(n_jobs=-1)(delayed(check_simplex)(simplex, space, alpha_shape, thresh) for simplex in simplices)
    mask = np.array(mask)
    # print(mask.sum())
    good_simplices = simplices[np.where(mask)]
    return good_simplices


all_vertices = np.zeros((0,3))
layer_height = .03
avg_vertex_dist = 20
layer_prune_rel = .15
prunable_layers = 5 # last n layers can be pruned, previous layers will be fixed
density_layers = 40 # number of layers to consider for calculating density

visualize = True
start_value, end_value = 571, 1079
length = len(layers[start_value:end_value])
triang_helper = np.array([[0., 0., -1e3]])

# make directories for the files containing the generated simplicial complexes
if not os.path.isdir(f'{object_name}_graphs'):
    os.mkdir(f'{object_name}_graphs')
if not os.path.isdir(f'{object_name}_graphs/layers_{start_value}_to_{start_value + length}'):
    os.mkdir(f'{object_name}_graphs/layers_{start_value}_to_{start_value + length}')

working_dir = f'{object_name}_graphs/layers_{start_value}_to_{start_value + length}'

# build the graphs by iterating over the layers and extending the previously generated ones
itr = tqdm(enumerate(layers[start_value:end_value]), total=len(layers[start_value:end_value]))
for i, (_, layer) in itr:
    with open(f'{object_name}/{layer}', 'rb') as f:
        times, layer_images, _ = pkl.load(f)

    # identify pixels that belong to the part and extract their spatial coordinates
    grid_vertices = identify_vertices(layer_images)
    grid_vertices_discrete = np.where(grid_vertices > .3, 1, 0)
    points = grid_vertices_to_vectors(grid_vertices, z=layer_height * i)

    # as a first step, prune in the 2-dimensional layer and append the remaining points to the 3d point cloud
    points = prune_space(points, int(np.round(layer_prune_rel * len(points))))
    all_vertices = np.concatenate([all_vertices, points], axis=0)

    # define which points can be pruned and which should be considered for the density calculation
    # in both cases, only consider a number of recent layers (but not the surface layer), s.t. the lower part of the point cloud is fixed
    prunable_vertices = np.where(np.logical_and(all_vertices[:, -1] > layer_height * (i - prunable_layers),
                                                all_vertices[:, -1] <= layer_height * (i - 1)), True, False)
    density_vertices = np.where(np.logical_and(all_vertices[:, -1] > layer_height * (i - density_layers),
                                               all_vertices[:, -1] <= layer_height * (i - 1)), True, False)

    # apply the selective pruning operation
    all_vertices = prune_selective(all_vertices, prunable_vertices, density_vertices, avg_vertex_dist)

    # generate the simplicial 3-complex using Delaunay-triangulation
    triangulation = Delaunay(np.concatenate([triang_helper, all_vertices], axis=0))
    simplices = [[v - 1 for v in simplex] for simplex in triangulation.simplices]
    additional_triangles = []
    for k, simplex in reversed(list(enumerate(simplices))):
        if -1 in simplex:
            additional_triangles.append(simplices.pop(k))
    simplices = np.array(simplices)

    if i > 0:  # use alphashape to estimate the hull of the part and find the simplices that are inside the hull
        try:
            alpha_shape = alphashape.alphashape(all_vertices, 2e-2 * np.sqrt(i + 1))
            boundary_points = alpha_shape.vertices
            good_simplices = get_good_simplices_parallel(simplices, all_vertices, alpha_shape)
        except Exception as e:
            print_exception(e)
            good_simplices = simplices
            boundary_points = all_vertices
    else:
        good_simplices = additional_triangles
        for simplex in good_simplices:
            for k, v in reversed(list(enumerate(simplex))):
                if v == -1:
                    simplex.pop(k)
        boundary_points = all_vertices

    # save the generated simplicial complex
    with open(os.path.join(working_dir, f'layer_{i}.pkl'), 'wb') as f:
        pkl.dump((all_vertices, good_simplices, boundary_points), f)
    itr.set_postfix({'vertices': len(all_vertices), 'top_vertices': len(points)})

    # visualize point cloud, simplicial complex and alphashape
    if visualize and i % 20 == 0:
        #matplotlib
        #inline
        if i == 0:
            good_triangles = good_simplices
        else:
            good_triangles = np.concatenate([
                good_simplices[:, (0, 1, 2)],
                good_simplices[:, (0, 1, 3)],
                good_simplices[:, (0, 2, 3)],
                good_simplices[:, (1, 2, 3)],
            ])

        """fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.scatter(all_vertices[:, 0], all_vertices[:, 1], all_vertices[:, 2])

        ax.set_xlim(0, 10)  # Set the range for the x-axis
        ax.set_ylim(0, 20)  # Set the range for the y-axis
        ax.set_zlim(0, 0.001)  # Set the range for the z-axis

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.plot_trisurf(all_vertices[:, 0], all_vertices[:, 1], all_vertices[:, 2], triangles=good_triangles, alpha=.1)
        ax.set_xlim(0, 10)  # Set the range for the x-axis
        ax.set_ylim(0, 20)  # Set the range for the y-axis
        ax.set_zlim(0, 0.001)  # Set the range for the z-axis

        if i > 0:
            ax = fig.add_subplot(1, 3, 3, projection='3d')
            ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces)

        # Set axis ranges
        ax.set_xlim(0, 10)  # Set the range for the x-axis
        ax.set_ylim(0, 20)  # Set the range for the y-axis
        ax.set_zlim(0, 0.001)  # Set the range for the z-axis

        # ax.set(xticklabels=[],  #Optionally remove tick labels
        #      yticklabels=[],
        #     zticklabels=[])

        # plt.show()
        plt.savefig(f'Benchmark_Architekture/Graph_layer_{i}.png', bbox_inches='tight')
        if i % 20 == 0:
            plt.close('all')

    # matplotlib tk
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(all_vertices[:, 0], all_vertices[:, 1], all_vertices[:, 2])
    plt.show()

    good_triangles = np.concatenate([
        good_simplices[:, (0, 1, 2)],
        good_simplices[:, (0, 1, 3)],
        good_simplices[:, (0, 2, 3)],
        good_simplices[:, (1, 2, 3)],
    ])

    # matplotlib tk
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(all_vertices[:, 0], all_vertices[:, 1], all_vertices[:, 2], triangles=good_triangles, alpha=.1)
    plt.show()

    alpha_shape.show()

    graph_edges_ixs = np.concatenate([
        good_triangles[:, (0, 1)],
        good_triangles[:, (0, 2)],
        good_triangles[:, (1, 2)],
    ])

    graph_edges = all_vertices[graph_edges_ixs[:, 0]], all_vertices[graph_edges_ixs[:, 1]]

    # # plot graph edges
    # %matplotlib tk
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for v1,v2 in zip(*graph_edges):
    #   ax.plot(*zip(v1,v2))
    #  plt.show()"""