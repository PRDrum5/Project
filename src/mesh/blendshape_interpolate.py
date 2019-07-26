import numpy as np
import os 
from plyfile import PlyData
from mesh import Mesh
from mesh import gen_file_list

dir_plys = 'root_meshes/voca_root/'
dir_path = os.path.dirname(os.path.realpath(__file__))

ply_path = os.path.join(dir_path, dir_plys)

f_list = gen_file_list(ply_path)

mesh = Mesh(f_list)
mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
mesh.get_vertex_postions(mesh_vertices, no_progress=True)
mesh_vertices = mesh.vertices_to_2d(mesh_vertices)
verts = mesh_vertices[:,0]

shape_axis = np.load(os.path.join(dir_path, 
                                  'blendshape_axis/shape_axis_lrw_ff_100.npy'))
total_shapes = shape_axis.shape[1]
n_shapes = 100
shape_axis = np.delete(shape_axis, range(n_shapes, total_shapes), axis=1)

save_path = os.path.join(dir_path, 'axis1_interpolation')
if not os.path.exists(save_path):
    os.makedirs(save_path)

axis = 4
shape = np.reshape(shape_axis[:,axis], verts.shape)
step = 0.03
file_num = 0

for i in range(0, -20, -1):
    filename = 'face' + '%05d' % file_num
    path = os.path.join(save_path, filename)
    weight = step * i
    blend =  verts + (weight * shape)
    mesh.export_mesh(blend, mesh.mesh_connections, path)
    file_num += 1
