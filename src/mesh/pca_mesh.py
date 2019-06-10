from plyfile import PlyData
from sklearn.decomposition import PCA
import os
import numpy as np
import time

class Mesh_Collect():
    def __init__(self, ply_files):
        self.ply_files = ply_files
        self._num_vertices = 5023
        self.num_files = 0

    def ply_verts_to_array(self, ply_verts):
        length = ply_verts.count
        verts = np.empty((length, 3))

        for vert in range(length):
            v = ply_verts[vert]
            (x, y, z) = (v[t] for t in ('x', 'y', 'z'))
            verts[vert] = [x, y, z]

        return verts

    def list_of_plys(self):
        """
        Returns a list of ply files in given directory
        """
        files = []
        for file in os.listdir(self.ply_files):
            if file.endswith(".ply"):
                files.append(file)
        
        self.num_files = len(files)
        return files

    def collect_ply_data(self, ply_verts, target_array, array_postion):
        for vert in range(self._num_vertices):
            v = ply_verts[vert]
            (x, y, z) = (v[t] for t in ('x', 'y', 'z'))
            target_array[array_postion+vert] = [x, y, z]

ply_files = "/home/peter/Documents/Uni/Project/datasets/registereddata/FaceTalk_170725_00137_TA/sentence01"

ms = Mesh_Collect(ply_files)

files = ms.list_of_plys()

verts = np.zeros((ms._num_vertices * ms.num_files, 3))
position = 0
for file in files:
    filepath = os.path.join(ply_files, file)

    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
        ms.collect_ply_data(plydata['vertex'], verts, position)
    position += ms._num_vertices

pca = PCA(n_components=3)
pca.fit(verts)  

print(pca.explained_variance_ratio_)  
print(pca.singular_values_) 
print(pca.components_)