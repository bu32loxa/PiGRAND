PiGRAND: Physics-informed Graph Neural Diffusion 
A comprehensive understanding of heat transport is essential for optimizing various mechanical and engineering applications, including 3D printing. Recent advances in machine learning, combined with physics-based models, have enabled a powerful fusion of numerical methods and data-driven algorithms. This progress is driven by the availability of limited sensor data in various engineering and scientific domains, where the cost of data collection and the inaccessibility of certain measurements are high. 
To this end, we present PiGRAND, a Physics-informed graph neural diffusion framework. In order to reduce the computational complexity of graph learning, an efficient graph construction procedure was developed. Our approach is inspired by the explicit Euler and implicit Crank-Nicolson methods for modeling continuous heat transport, leveraging sub-learning models to secure the accurate diffusion across graph nodes. To enhance computational performance, our approach is combined with efficient transfer learning. We evaluate PiGRAND on thermal images from 3D printing, demonstrating significant improvements in prediction accuracy and computational performance compared to traditional graph neural diffusion (GRAND) and physics-informed neural networks (PINNs). These enhancements are attributed to the incorporation of physical principles derived from the theoretical study of partial differential equations (PDEs) into the learning model. 
http://arxiv.org/abs/2603.15194

# Graph-based diffusion model for predicting internal heat distribution

## extract layer information from images

Run 'LayerSeperation_2.py' to extract the individual layers from the sequence of images.

## build simplicial complex

Executing the cells of 'build_graph_iterative.ipynb' creates a series of simplicial complexes which model the partially printed object up to the current layer. After creating the simplicial complexes, run 'adjacencies_boundary.py' to extract the underlying graphs and to identify the vertex classes.

## fit diffusion model

Having created the graph representations, run 'fit_diffusion_model.ipynb' to fit an explicit model (based on a forward Euler step) or 'fit_implicit_model.ipynb' to fit an implicit model (based on Crank-Nicolson timesteps).

