import os
import numpy as np
from mesh import Mesh
from mesh import gen_file_list
from tqdm import tqdm

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    shapes = np.load(os.path.join(dir_path, 'root_diff_shape_axis_10.npy'))
    total_shapes = shapes.shape[1]
    n_shapes = 10 # Number of principle axis to use.
    shapes = np.delete(shapes, range(n_shapes, total_shapes), axis=1)

    for file in tqdm(range(40)):
        sentence = 'sentence' + '%02d' % (file+1)
        dir_plys = 'aligned/Subject01/'

        # instantiate Mesh class with lsit of ply files
        f_list = gen_file_list(os.path.join(dir_path, dir_plys, sentence))
        mesh = Mesh(f_list)

        ## Overwrite default root mesh
        #mesh.root_mesh = root_mesh.root_mesh

        mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
        mesh.get_vertex_postions(mesh_vertices)
        mesh_vertices = mesh.vertices_to_2d(mesh_vertices)

        params = np.empty((n_shapes, mesh.num_files))
        for v in range(mesh.num_files):
            params[:,v] = mesh.recover_blendshape_parameters(
                mesh_vertices[:,v], shapes)
        
        save_dir = 'shape_params_' + str(n_shapes)
        save_path = os.path.join(dir_path, save_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_save_path = os.path.join(save_path, sentence)
        np.save(file_save_path, params)