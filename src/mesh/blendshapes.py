from plyfile import PlyData, PlyElement
import scipy.linalg as la
import os
import numpy as np
from random import shuffle

class Blendshapes():
    def __init__(self, ply_files=None):
        self.ply_files = ply_files
        self._num_vertices = 5023
        self.num_files = 1

    def list_of_plys(self, keep=1):
        """
        Returns a list of ply files in given directory
        If keep is less than 1, that proportion of files are kept.
        """
        selected_files = []
        for root, _dirs, files in os.walk(self.ply_files):
            for file in files:
                if file.endswith(".ply"):
                    selected_files.append(os.path.join(root, file))
        
        self.num_files = int(round(keep * len(selected_files)))
        shuffle(selected_files)
        del selected_files[self.num_files:]
        
        return selected_files

    def collect_ply_data(self, ply_verts, target_array, sample_number=0):
        index = 0
        for vert in range(self._num_vertices):
            v = ply_verts[vert]
            (x, y, z) = (v[t] for t in ('x', 'y', 'z'))
            target_array[index][sample_number] = x
            target_array[index + 1][sample_number] = y
            target_array[index + 2][sample_number] = z
            index += 3
    
    def collect_mesh_faces(self, ply_file):
        with open(ply_file, 'rb') as f:
            plydata = PlyData.read(f)
            faces = plydata['face'].data
        return faces
    
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
    
    def export_mesh(self, verts, faces, filename=None, text=False):
        if filename == None:
            filename = 'my_plyfile'

        vertices = np.empty((self._num_vertices,), 
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        index = 0
        for vert in range(self._num_vertices):
            vertices[vert] = (verts[index], verts[index+1], verts[index+2])
            index += 3

        el_faces = PlyElement.describe(faces, 'face')
        el_vertices = PlyElement.describe(vertices, 'vertex')
        PlyData([el_vertices, el_faces], text=text).write(filename + '.ply') 

if __name__ == "__main__":

    ply_files = "/home/peter/Documents/Uni/Project/datasets/registereddata/FaceTalk_170725_00137_TA/sentence01"

    bs = Blendshapes(ply_files)

    files = bs.list_of_plys(keep=1)

    faces = bs.collect_mesh_faces(files[0])

    verts = np.empty((3 * bs._num_vertices, bs.num_files))

    sample_number = 0
    print("Number of files being processed: {}".format(bs.num_files))
    for file in files:
        filepath = os.path.join(ply_files, file)

        with open(filepath, 'rb') as f:
            plydata = PlyData.read(f)
            bs.collect_ply_data(plydata['vertex'], verts, sample_number)
        sample_number += 1

        if (sample_number % 100) == 0:
            print("Processing file number: {}".format(sample_number))

    #mean_verts = np.mean(verts, axis=1, keepdims=True)
    #bs.export_mesh(mean_verts, faces, filename='mean_mesh')

    shapes = bs.create_blendshapes(verts, 20)
    np.savetxt('blendshapes_sentence01.txt', shapes, delimiter=',')

    #export_mesh()

    #TODO write mean mesh to ply file.
    #TODO apply and interpolate blendshapes on mean mesh
