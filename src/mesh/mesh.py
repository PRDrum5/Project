from plyfile import PlyData, PlyElement
from sklearn.decomposition import IncrementalPCA, PCA
import scipy.linalg as la
import os
import numpy as np
from tqdm import tqdm

def gen_file_list(path, ext='.ply'):
    f_list = []
    for root, _dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                f_list.append(os.path.join(root, file))
    f_list = sorted(f_list)
    return f_list

class Mesh():
    def __init__(self, ply_files, save_path=None):
        self.save_path = save_path

        if self.save_path and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.ply_files = sorted(ply_files)
        self.num_files = len(self.ply_files)
        self._get_mesh_metadata(self.ply_files[0])
        self._get_root_mesh(self.ply_files[0])

    def _get_root_mesh(self, root_ply):
        self.root_mesh = self.get_empty_vertices(1)
        self.get_vertex_postions(self.root_mesh)

    def _get_mesh_metadata(self, example_ply):
        with open(example_ply, 'rb') as f:
            plydata = PlyData.read(f)
            self.vertex_count = plydata['vertex'].count
            self.mesh_connections = plydata['face'].data

    def get_empty_vertices(self, num_files, dtype=None):
        if dtype:
            mesh_vertices = np.empty((self.vertex_count, 3, num_files),                                  dtype=dtype)
        else:
            mesh_vertices = np.empty((self.vertex_count, 3, num_files))
        return mesh_vertices

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

        for file_number, file in tqdm(enumerate(self.ply_files)):
            if file_number == target_array.shape[2]:
                break
            with open(file, 'rb') as f:
                plydata = PlyData.read(f)
                self._get_vertex_postions(plydata, target_array, file_number)
    
    def vertices_to_2d(self, vertices):
        """
        Convert 3d tensor of vertices to 2d array.
        each column is a seperate mesh file
        input shape (n_vertices, 3, n_files)
        output shape (3*n_vertices, n_files)
        """

        return vertices.reshape(3*self.vertex_count, -1)
    
    def vertices_to_3d(self, vertices):
        """
        convert 2d array to 3d tensor.
        input shape (3*n_vertices, n_file)
        output shape (n_vertices, 3, n_file)
        """
        return vertices.reshape(self.vertex_count, 3, self.num_files)

    
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
    
    def export_all_meshes(self, vertices, filepath, filename):
        faces = self.mesh_connections

        for file in range(self.num_files):
            save_path = os.path.join(filepath, filename + '%05d' % file)
            self.export_mesh(vertices[:,file], faces, filename=save_path)
    
    def create_blendshapes(self, vertices, n_shapes=None):
        # perform PCA on vertices to obtrain blendshapes
        v = vertices.T @ vertices
        eigvals, eigvecs = la.eig(v)    # Eigen analysis
        eigvecs = np.real(eigvecs)
        eigvals = np.real(eigvals)

        # Remove zero eigen values
        zero_idx = np.where(eigvals < 10E-10)[0]
        eigvecs = np.delete(eigvecs, zero_idx, axis=1)
        eigvals = np.delete(eigvals, zero_idx)

        # Sort eigenvectors by eigenvalues
        idx = eigvals.argsort()[::-1]   
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]
        
        # Create diagonal of eigen values
        eigvals_diag = np.diag(eigvals)

        # Create the blendshapes transformation matrix
        blendshapes = vertices @ eigvecs @ la.inv(eigvals_diag)

        # Reduce down to amount desired
        if n_shapes:
            blendshapes = np.delete(blendshapes, np.s_[n_shapes:], axis=1)

        return blendshapes
    
    def inc_create_blendshapes(self, vertices, 
                               batch_size=1000, n_components=None):

        ipca = IncrementalPCA(n_components, whiten=True, batch_size=batch_size)
        ipca.fit(vertices)
        eigenvectors = ipca.components_

        return eigenvectors.T

    def new_create_blendshapes(self, vertices, n_components=None):
        pca = PCA(n_components=n_components, whiten=True)
        pca.fit(vertices)
        eigenvectors = pca.components_

        return eigenvectors.T
    
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
    
    def create_frame_zero_diff(self, vertices):
        """
        Subtracts each frame from the first frame, resulting in a motion flow 
        from the first frame.
        """
        first_frame = vertices[:,0].reshape(-1, 1)
        diff = vertices - first_frame
        return diff
    
    def create_given_frame_diff(self, vertices, root_vertices):
        """
        Subtracts each from from the given root mesh
        """
        diff = vertices - root_vertices
        return diff
    
    def _apply_procrustes(self, matrix1, matrix2):
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

        # Rotate matrix 2
        rotation, _ = la.orthogonal_procrustes(matrix1, matrix2)
        matrix2 = (matrix2 @ rotation.T)

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
        
        matrix1, matrix2 = self._apply_procrustes(matrix1, matrix2)

        diff = np.sum(np.square(matrix1 - matrix2))

        return matrix1, matrix2, diff
    
    def mesh_alignment(self, verts, root_mesh):
        landmarks = [3365, 2433, 3150, 1402, 1417, 4336, 4892]#, 338, 27]
        #[nose, out R eye, in R eye, out L eye, in L eye, back R eye, back L eye]#, R brow, L brow]
        mat1 = root_mesh[:,:,0]
        for mesh in range(0, self.num_files):
            mat2 = verts[:,:,mesh]
            mat1, mat2, diff = self.procrustes(mat1, mat2, landmarks)

    def morph_mesh(self, vertices, blendshapes, shape_params):
        """
        Given blendshape axis and parameters to move along, morphs the given 
        mesh vertices.
        """
        n_shapes = blendshapes.shape[1]
        if len(shape_params) < n_shapes:
            shape_params = np.pad(shape_params, (0,n_shapes), 'constant')
        
        morph = shape_params * blendshapes
        morph = np.sum(morph, axis=1).reshape(vertices.shape)

        morphed_vertices = vertices + morph
        morphed_vertices = morphed_vertices.reshape(-1,1)

        return morphed_vertices

    def recover_blendshape_parameters(self, altered_vertices, blendshapes):
        """
        Given an altered set of mesh vertices and the blendshape axis along
        which the mesh has been morphed, attempt to recover the parameters used
        to morph from the root mesh.
        """
        n_shapes = blendshapes.shape[1]
        altered_vertices = altered_vertices.reshape(-1,1)
        
        root_vertices = self.vertices_to_2d(self.root_mesh)
        vert_deltas = altered_vertices - root_vertices

        shape_inv = np.linalg.pinv(blendshapes)
        params = shape_inv @ vert_deltas

        return params.reshape(-1,)

    def get_sequence_blendshapes(self, seq_vertices, shapes):
        """
        Calculate the blendshapes which desicribe a given sequence.
        """
        n_seq = seq_vertices.shape[1]
        n_shapes = shapes.shape[1]

        seq_shapes = np.zeros((n_shapes, n_seq))

        for v in range(n_seq):
            verts = seq_vertices[:,v]
            seq_shapes[:,v] = self.recover_blendshape_parameters(verts, shapes)
        
        return seq_shapes
    
    def export_sequence_blendshapes(self, seq_vertices, shapes, filename):
        """
        Save the blendshapes which describe a given sequence.
        """
        save_path = os.path.join(self.save_path, filename)
        seq_shapes = self.get_sequence_blendshapes(seq_vertices, shapes)
        np.save(save_path, seq_shapes)


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))

    ply_path = 'sentence01'
    blendshape_path = 'sequence_shapes'

    ply_files = os.path.join(dir_path, ply_path)
    save_path = os.path.join(dir_path, blendshape_path)

    mesh = Mesh(ply_files, save_path)

    mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
    mesh.get_vertex_postions(mesh_vertices)
    mesh_vertices = mesh.vertices_to_2d(mesh_vertices)

    shapes = np.load(os.path.join(dir_path, 'shapes00.npy'))
    total_shapes = shapes.shape[1]
    n_shapes = 10
    shapes = np.delete(shapes, range(n_shapes, total_shapes), axis=1)

    mesh.export_sequence_blendshapes(mesh_vertices, shapes, ply_path)
    