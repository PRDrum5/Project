import numpy as np
import os
import argparse
from plyfile import PlyData
from mesh.mesh import Mesh, gen_file_list

parser = argparse.ArgumentParser(description='Gan Output Interpolation')

parser.add_argument('--root_mesh', default='model/data/root_mesh', help='Path to root mesh')
parser.add_argument('--blendshape_axis', default='model/data/blendshape_axis/shape_axis.npy', help='Path to blendshape axis')
parser.add_argument('--params', default='', help='Path to params')
parser.add_argument('--out_path', default='gan_visualisation', help='Output path')

args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))

#root_mesh_dir = 'model/data/root_mesh'
root_mesh_dir = args.root_mesh

#blendshape_axis_dir = 'model/data/blendshape_axis/shape_axis.npy'
blendshape_axis_dir = args.blendshape_axis

#blendshape_param_dir = 'mesh/shape_params_5/sentence01.npy'
blendshape_param_dir = args.params

#save_dir = 'gan_visualisation'
save_dir = args.out_path

root_mesh_path = os.path.join(dir_path, root_mesh_dir)
blendshape_axis_path = os.path.join(dir_path, blendshape_axis_dir)
blendshape_param_path = os.path.join(dir_path, blendshape_param_dir)
save_path = os.path.join(dir_path, save_dir)
if not os.path.exists(save_path):
    os.makedirs(save_path)


## Load the root mesh to interpolate from
# Instantiate root mesh class
root_mesh_list = gen_file_list(root_mesh_path)
root_mesh = Mesh(root_mesh_list)

# Collect the vertices for the root mesh, and populate it
root_vertices = root_mesh.get_empty_vertices(root_mesh.num_files)
root_mesh.get_vertex_postions(root_vertices)

# Convert the vertices into 2d representation
root_vertices = root_mesh.vertices_to_2d(root_vertices)


## Load the blendshape axis to interpolate along
blendshape_axis = np.load(blendshape_axis_path)
total_axis = blendshape_axis.shape[1]

## Load generated blendshape parameters
shape_params = np.load(blendshape_param_path)

params_used = shape_params.shape[0]
shape_params = np.delete(shape_params, range(1, params_used), axis=0)
params_used = shape_params.shape[0]

# Remove blendshape axis not used.
params_not_used = range(params_used, total_axis)
blendshape_axis = np.delete(blendshape_axis, params_not_used, axis=1)


## Interpolate and export the root mesh for each frame in params
for frame in range(shape_params.shape[1]):
    filename = 'face' + '%03d' % frame
    export_path = os.path.join(save_path, filename)
    frame_params = shape_params[:, frame].reshape(-1,1)
    interpolation = blendshape_axis @ frame_params
    interpolated_vertices = root_vertices + interpolation
    root_mesh.export_mesh(interpolated_vertices, 
                          root_mesh.mesh_connections, 
                          export_path)
