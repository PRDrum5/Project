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
        mesh_vertices = np.empty((self.vertex_count, 3, num_files))
        return mesh_vertices

    def old_get_vertex_postions(self, plydata, target_array, file_number=0):
        index = 0
        for vert in range(self.vertex_count):
            v = plydata['vertex'][vert]
            target_array[index][file_number] = v[0]
            target_array[index+1][file_number] = v[1]
            target_array[index+2][file_number] = v[2]
            index += 3
        return target_array
    
    def _get_vertex_postions(self, plydata, target_array, file_number=0):
        for vert in range(self.vertex_count):
            v = plydata['vertex'][vert]
            target_array[vert][0][file_number] = v[0]
            target_array[vert][1][file_number] = v[1]
            target_array[vert][2][file_number] = v[2]
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
    
    def veticies_to_2d(self, vertices):
        """
        Convert 3d tensor of vertices to 2d array.
        each column is a seperate mesh file
        input shape (n_vertices, 3, n_files)
        output shape (3*n_vertices, n_files)
        """

        return vertices.reshape(3*mesh.vertex_count, mesh.num_files)
    
    def vertices_to_3d(self, vertices):
        """
        convert 2d array to 3d tensor.
        input shape (3*n_vertices, n_file)
        output shape (n_vertices, 3, n_file)
        """
        return vertices.reshape(mesh.vertex_count, 3, mesh.num_files)

    
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
    
    def _apply_procrustes(self, matrix1, matrix2, params):
        """
        params = [mean_matrix1, 
                  mean_matrix2, 
                  norm_matrix1, 
                  norm_matrix2,
                  rotation_matrix,
                  scaling]
        """
        matrix1 -= params[0]
        matrix2 -= params[1]
        matrix1 /= params[2]
        matrix2 /= params[3]
        matrix2 = (matrix2 @ params[4].T) * params[5]

        return matrix1, matrix2
    
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
        matrix1, matrix2 = self._apply_procrustes(matrix1, matrix2, params)

        diff = np.sum(np.square(matrix1 - matrix2))

        return matrix1, matrix2, diff
    
    def mesh_alignment(self, verts):
        landmarks = [0, 1, 2, 3]
        mat1 = verts[:,:,0]
        for mesh in range(1, self.num_files):
            mat2 = verts[:,:,mesh]
            mat1, mat2, diff = self.procrustes(mat1, mat2, landmarks)

    
if __name__ == "__main__":

    dir_plys = "/home/peter/Documents/Uni/Project/datasets/registereddata/FaceTalk_170725_00137_TA/sentence01"

    mesh = Mesh(dir_plys)
    mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
    mesh.get_vertex_postions(mesh_vertices)

    print(mesh_vertices[10][0][0])
    mesh.mesh_alignment(mesh_vertices)
    print(mesh_vertices[10][0][0])
#    mat1, mat2, diff = mesh.procrustes(a, b, [0,1,2,3])

    mesh_vertices = mesh.veticies_to_2d(mesh_vertices)
    frame_deltas = mesh.create_frame_deltas(mesh_vertices)
    shapes = mesh.create_blendshapes(frame_deltas, 3)

    mesh.export_mesh(mesh_vertices[:,0], mesh.mesh_connections, filename='temp')