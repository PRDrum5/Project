import os
import numpy as np
import argparse
from mesh import Mesh
from mesh import gen_file_list
from tqdm import tqdm

def recover_for_subject(subject, subject_path):
    """
    Given a path to a subject directory, 
    recover the blendshapes for all sentences under that subject path.
    """
    sentence_list = sorted(os.listdir(subject_path))
    for sentence_num, sentence in enumerate(sentence_list):
        ply_files = os.path.join(subject_path, sentence)
        params_dir = args.params_dir
        params_name = args.params_name
        save_dir = os.path.join(params_dir, 
                                params_name + '_' + str(n_shapes),
                                subject)
        save_path = os.path.join(dir_path, save_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_save_path = os.path.join(save_path, sentence)
        if os.path.exists(file_save_path + '.npy'):
            continue

        # instantiate Mesh class with lsit of ply files
        f_list = gen_file_list(ply_files)
        mesh = Mesh(f_list)

        # Overwrite default root mesh
        mesh.root_mesh = root_mesh.root_mesh
        mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
        mesh.get_vertex_postions(mesh_vertices, no_progress=True)
        mesh_vertices = mesh.vertices_to_2d(mesh_vertices)
        params = np.empty((n_shapes, mesh.num_files))
        for v in range(mesh.num_files):
            params[:,v] = mesh.recover_blendshape_parameters(
                mesh_vertices[:,v], shapes)

        np.save(file_save_path, params)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recover Blendshape Params')
    parser.add_argument('--blendshape_axis_dir', default='blendshape_axis/')
    parser.add_argument('--axis_file', default='shape_axis_100.npy')
    parser.add_argument('--n_params', default='2')
    parser.add_argument('--params_dir', default='shape_params')
    parser.add_argument('--params_name', default='shape_params')
    parser.add_argument('--root_mesh_dir', 
                        default='root_meshes/voca_root/')
    parser.add_argument('--recover_from_dir', default='all_aligned/')

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

    dir_plys = args.recover_from_dir
    all_subjects_path = os.path.join(dir_path, dir_plys)
    subjects_list = sorted(os.listdir(all_subjects_path))

    for subject in tqdm(subjects_list):
        subject_path = os.path.join(all_subjects_path, subject)
        recover_for_subject(subject, subject_path)
