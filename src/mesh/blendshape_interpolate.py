import numpy as np
import os 
from plyfile import PlyData
from mesh import Mesh
from mesh import gen_file_list

dir_plys = 'aligned/sentence01'
dir_path = os.path.dirname(os.path.realpath(__file__))

ply_path = os.path.join(dir_path, dir_plys)

f_list = gen_file_list(ply_path)

mesh = Mesh(ply_path)
mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
mesh.get_vertex_postions(mesh_vertices)
mesh_vertices = mesh.vertices_to_2d(mesh_vertices)
verts = mesh_vertices[:,0]

shapes = np.load(os.path.join(dir_path, 'shapes00.npy'))
total_shapes = shapes.shape[1]
n_shapes = 100
shapes = np.delete(shapes, range(n_shapes, total_shapes), axis=1)

save_path = os.path.join(dir_path, 'blendshapes/shape100')
if not os.path.exists(save_path):
    os.makedirs(save_path)

std_along_shapes = 'aligned/standard_div_subject1.npy'
std = np.load(os.path.join(dir_path, std_along_shapes))

axis = 99
shape = np.reshape(shapes[:,axis], verts.shape)
step = 0.0005
axis_std = std[axis]
file_num = 0

for i in range(-100, 100, 1):
    filename = 'face' + '%05d' % file_num
    path = os.path.join(save_path, filename)
    weight = axis_std * step * i
    blend =  verts + (weight * shape)
    mesh.export_mesh(blend, mesh.mesh_connections, path)
    file_num += 1
