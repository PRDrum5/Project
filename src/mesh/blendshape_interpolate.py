import numpy as np
import os 
from plyfile import PlyData
from blendshapes import Blendshapes
from sklearn.preprocessing import normalize

dir_path = os.path.dirname(os.path.realpath(__file__))
ply_file = os.path.join(dir_path, '/home/peter/Documents/Uni/Project/datasets/registereddata/FaceTalk_170725_00137_TA/sentence01/sentence01.000001.ply')
shapes_file = os.path.join(dir_path, 'blendshapes_sentence01.txt')

bs = Blendshapes(ply_files=ply_file)

verts = np.empty((3 * bs._num_vertices, bs.num_files))

with open(bs.ply_files, 'rb') as f:
    plydata = PlyData.read(f)
    bs.collect_ply_data(plydata['vertex'], verts)
    faces = plydata['face'].data

shapes = np.loadtxt(shapes_file, delimiter=',')
shapes = normalize(shapes, axis=1)

#princial_0 = verts + np.reshape(0.05 * shapes[:,0], verts.shape)
#
#bs.export_mesh(princial_0, faces, 'princ3')

save_path = os.path.join(dir_path, 'blendshapes/shape00')

shape = np.reshape(shapes[:,0], verts.shape)
step = 0.001
file_num = 0
for i in range(-50, 51, 1):
    filename = 'face' + '%05d' % file_num
    path = os.path.join(save_path, filename)
    weight = step * i
    blend = verts + (weight * shape)
    bs.export_mesh(blend, faces, path)
    file_num += 1
