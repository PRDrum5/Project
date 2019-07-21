import os
import numpy as np
import argparse
from mesh import Mesh
from mesh import gen_file_list
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recover Blendshape Params')
    parser.add_argument('--blendshape_axis_dir', default='blendshape_axis/')
    parser.add_argument('--axis_file', default='shape_axis_100.npy')
    parser.add_argument('--n_params', default='10')
    parser.add_argument('--params_dir', default='shape_params')
    parser.add_argument('--params_name', default='shape_params')
    parser.add_argument('--root_mesh_dir', default='root_meshes/Subject01/')
    parser.add_argument('--recover_from_dir', default='aligned/Subject01/')

    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    shapes_axis_dir = args.blendshape_axis_dir
    shapes_file = args.axis_file

    shapes = np.load(os.path.join(dir_path, shapes_axis_dir, shapes_file))

    total_shapes = shapes.shape[1]
    n_shapes = int(args.n_params) # Number of principle axis to use.
    shapes = np.delete(shapes, range(n_shapes, total_shapes), axis=1)

    root_mesh_dir = args.root_mesh_dir
    root_mesh_path = os.path.join(dir_path, root_mesh_dir)
    root_mesh_list = gen_file_list(root_mesh_path, ext='.ply')
    root_mesh = Mesh(root_mesh_list)

    #TODO expand this to work with any number of nested sentences
    for file in tqdm(range(1)):
        sentence = 'sentence' + '%02d' % (file+1)
        dir_plys = args.recover_from_dir

        # instantiate Mesh class with lsit of ply files
        f_list = gen_file_list(os.path.join(dir_path, dir_plys, sentence))
        mesh = Mesh(f_list)

        # Overwrite default root mesh
        mesh.root_mesh = root_mesh.root_mesh

        mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
        mesh.get_vertex_postions(mesh_vertices)
        mesh_vertices = mesh.vertices_to_2d(mesh_vertices)

        params = np.empty((n_shapes, mesh.num_files))
        for v in range(mesh.num_files):
            params[:,v] = mesh.recover_blendshape_parameters(
                mesh_vertices[:,v], shapes)
        
        params_dir = args.params_dir
        params_name = args.params_name
        save_dir = os.path.join(params_dir, params_name + '_' + str(n_shapes))
        save_path = os.path.join(dir_path, save_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_save_path = os.path.join(save_path, sentence)
        np.save(file_save_path, params)