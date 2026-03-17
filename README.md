# Graph-based diffusion model for predicting internal heat distribution

## extract layer information from images

Run 'LayerSeperation_2.py' to extract the individual layers from the sequence of images.

## build simplicial complex

Executing the cells of 'build_graph_iterative.ipynb' creates a series of simplicial complexes which model the partially printed object up to the current layer. After creating the simplicial complexes, run 'adjacencies_boundary.py' to extract the underlying graphs and to identify the vertex classes.

## fit diffusion model

Having created the graph representations, run 'fit_diffusion_model.ipynb' to fit an explicit model (based on a forward Euler step) or 'fit_implicit_model.ipynb' to fit an implicit model (based on Crank-Nicolson timesteps).