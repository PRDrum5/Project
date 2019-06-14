import numpy as np
import os 
from plyfile import PlyData
from mesh import Mesh
from sklearn.preprocessing import normalize

dir_plys = 'sentence01'
dir_path = os.path.dirname(os.path.realpath(__file__))
mesh = Mesh(os.path.join(dir_path, dir_plys))
mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
mesh.get_vertex_postions(mesh_vertices)
mesh_vertices = mesh.vertices_to_2d(mesh_vertices)
verts = mesh_vertices[:,0]

shapes = np.loadtxt(os.path.join(dir_path, 'shapes00.txt'), delimiter=',')
shapes = normalize(shapes, axis=1)

save_path = os.path.join(dir_path, 'blendshapes/shape01')
if not os.path.exists(save_path):
    os.makedirs(save_path)

shape = np.reshape(shapes[:,1], verts.shape)
step = 0.001
file_num = 0
for i in range(-100, 101, 1):
    filename = 'face' + '%05d' % file_num
    path = os.path.join(save_path, filename)
    weight = step * i
    blend =  verts + (weight * shape)
    mesh.export_mesh(blend, mesh.mesh_connections, path)
    file_num += 1
