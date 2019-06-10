from plyfile import PlyData
import scipy.linalg as la
import os
import numpy as np
import time

class Blendshapes():
    def __init__(self, ply_files):
        self.ply_files = ply_files
        self._num_vertices = 5023
        self.num_files = 0

    def list_of_plys(self):
        """
        Returns a list of ply files in given directory
        """
        selected_files = []
        for root, _dirs, files in os.walk(self.ply_files):
            for file in files:
                if file.endswith("1.ply"):
                    selected_files.append(os.path.join(root, file))
        
        self.num_files = len(selected_files)
        return selected_files

    def collect_ply_data(self, ply_verts, target_array, sample_number):
        index = 0
        for vert in range(self._num_vertices):
            v = ply_verts[vert]
            (x, y, z) = (v[t] for t in ('x', 'y', 'z'))
            target_array[index][sample_number] = x
            target_array[index + 1][sample_number] = y
            target_array[index + 2][sample_number] = z
            index += 3
    
    def create_blendshapes(self, vertices, n_shapes):
        # Subtract the mean
        vertices = vertices - np.mean(vertices, axis=1, keepdims=True)
        v = vertices.T @ vertices

        # Eigen Analysis and sorting 
        eigvals, eigvecs = la.eig(v)
        idx = eigvals.argsort()[::-1]   
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]

        eigvals_diag = np.diag(np.diag(eigvecs))

        transform = vertices @ eigvecs @ la.inv(eigvals_diag)
        blendshapes = np.delete(transform, np.s_[n_shapes:], axis=1)

        return blendshapes


ply_files = "/home/peter/Documents/Uni/Project/datasets/registereddata/"#FaceTalk_170725_00137_TA"#/sentence01"

bs = Blendshapes(ply_files)

files = bs.list_of_plys()

verts = np.zeros((3 * bs._num_vertices, bs.num_files))
sample_number = 0
for file in files:
    filepath = os.path.join(ply_files, file)

    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
        bs.collect_ply_data(plydata['vertex'], verts, sample_number)
    sample_number += 1

shapes = bs.create_blendshapes(verts, 10)
np.savetxt('blendshapes_1.txt', shapes, delimiter=',')
