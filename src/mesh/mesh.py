from plyfile import PlyData, PlyElement
import scipy.linalg as la
import os
import numpy as np

class Mesh():
    def __init__(self, dir_plys, mesh_deltas=False):
        self.dir_plys = dir_plys

        self.ply_files = []

        for root, _dirs, files in os.walk(self.dir_plys):
            for file in files:
                if file.endswith("01.ply"):
                    self.ply_files.append(os.path.join(root, file))
        
        self.ply_files = sorted(self.ply_files)
        self.num_files = len(self.ply_files)
        self._get_mesh_metadata(self.ply_files[0])

    def _get_mesh_metadata(self, example_ply):
        with open(example_ply, 'rb') as f:
            plydata = PlyData.read(f)
            self.vertex_count = plydata['vertex'].count
            self.mesh_connections = plydata['face'].data

    def get_empty_vertices(self, num_files):
        mesh_vertices = np.empty((3*self.vertex_count, num_files))
        return mesh_vertices

    def _get_vertex_postions(self, plydata, target_array, file_number=0):
        index = 0
        for vert in range(self.vertex_count):
            v = plydata['vertex'][vert]
            (x, y, z) = (v[t] for t in ('x', 'y', 'z'))
            target_array[index][file_number] = x
            target_array[index+1][file_number] = y
            target_array[index+2][file_number] = z
            index += 3
        return target_array

    def get_vertex_postions(self, target_array):
        """
        Extracts vertex postions for all ply files
        """
        print("Number of files being processed: {}".format(self.num_files))

        for file_number, file in enumerate(self.ply_files):
            with open(file, 'rb') as f:
                plydata = PlyData.read(f)
                self._get_vertex_postions(plydata, target_array, file_number)
                if (file_number % 10) == 0:
                    print("Processing file number: {}".format(file_number))
    
    def export_mesh(self, mesh_vertices, mesh_connections, 
                    filename=None, text=False):
        """
        Given numpy array of xyz postions for each vertex, export mesh back
        to a ply file with the vertex connections from original imported files.
        """
        if filename == None:
            filename = 'my_plyfile'

        vertices = np.empty((self.vertex_count,), 
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        index = 0
        for vert in range(self.vertex_count):
            vertices[vert] = (mesh_vertices[index], 
                              mesh_vertices[index+1], 
                              mesh_vertices[index+2])
            index += 3

        el_faces = PlyElement.describe(mesh_connections, 'face')
        el_vertices = PlyElement.describe(vertices, 'vertex')
        PlyData([el_vertices, el_faces], text=text).write(filename + '.ply')
    
    def create_blendshapes(self, vertices, n_shapes):
        # perform PCA on vertices to obtrain blendshapes
        v = vertices.T @ vertices
        eigvals, eigvecs = la.eig(v)    # Eigen analysis

        # Sort eigenvectors by eigenvalues
        idx = eigvals.argsort()[::-1]   
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]
        
        # Create diagonal of eigen values
        eigvals_diag = np.diag(np.diag(eigvecs))

        # Create the blendshapes transformation matrix
        transform = vertices @ eigvecs @ la.inv(eigvals_diag)

        # Reduce down to amount desired
        blendshapes = np.delete(transform, np.s_[n_shapes:], axis=1)

        return blendshapes
    
    def create_frame_deltas(self, vertices):
        """
        Subtracts each subsequent frame resulting in the motion flow of the mesh
        """
        # Need an even number of samples
        if (vertices.shape[1] % 2) != 0:
            vertices = np.delete(vertices, -1, 1)
        
        # even columns - odd columns
        vertices = vertices[:,1::2] - vertices[:,::2]
        return vertices
    
    def _get_procrustes_parameters(self, matrix1, matrix2):
        # Translate centroids of matricies to origin
        mean1 = np.mean(matrix1, 0)
        mean2 = np.mean(matrix2, 0)
        matrix1 -= mean1
        matrix2 -= mean2

        # Scale the matrix the same
        norm1 = la.norm(matrix1)
        norm2 = la.norm(matrix2)
        matrix1 /= norm1
        matrix2 /= norm2

        rotation, scale = la.orthogonal_procrustes(matrix1, matrix2)

        return mean1, mean2, norm1, norm2, rotation, scale
    
    def procrustes(self, matrix1, matrix2, landmarks):
        """
        Given two matrices of equal shape, bring them into alignment about 
        given landmarks.
        """
        n_landmarks = len(landmarks)
        dim = matrix1.shape[1]
        submat1 = np.zeros((n_landmarks, dim))
        submat2 = np.zeros((n_landmarks, dim))

        for i, landmark_idx in enumerate(landmarks):
            submat1[i] = matrix1[landmark_idx]
            submat2[i] = matrix2[landmark_idx]
        
        params = self._get_procrustes_parameters(submat1, submat2)
        mean_sub1 = params[0]
        mean_sub2 = params[1]
        norm1 = params[2]
        norm2 = params[3]
        rotation_matrix = params[4]
        scale = params[5]
        print(params[0])
        print(params[1])
        print(params[2])
        print(params[3])
        print(params[4])
        print(params[5])
    
if __name__ == "__main__":

    dir_plys = "/home/peter/Documents/Uni/Project/datasets/registereddata/FaceTalk_170725_00137_TA/sentence01"

    mesh = Mesh(dir_plys)
    a = np.random.randn(5,3)
    b = np.random.randn(5,3)
    mesh.procrustes(a, b, [0,1,2])
    #mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
    #mesh.get_vertex_postions(mesh_vertices)
    #frame_deltas = mesh.create_frame_deltas(mesh_vertices)
    #shapes = mesh.create_blendshapes(frame_deltas, 3)

    #mesh.export_mesh(mesh_vertices[:,0], mesh.mesh_connections, filename='temp')