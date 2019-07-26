import os
import numpy as np
from mesh import Mesh
from mesh import gen_file_list
from tqdm import tqdm

if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Import root mesh to align onto
    root_mesh_dir = 'root_meshes/voca_root'
    #root_mesh_dir = 'root_meshes/Subject01_aligned'
    root_mesh_path = os.path.join(dir_path, root_mesh_dir)

    root_mesh_list = gen_file_list(root_mesh_path, ext='.ply')

    root_mesh = Mesh(root_mesh_list)
    root_mesh_vertices = root_mesh.get_empty_vertices(root_mesh.num_files)
    root_mesh.get_vertex_postions(root_mesh_vertices, no_progress=True)

    unaligned_dir = 'unaligned/'
    unaligned_path = os.path.join(dir_path, unaligned_dir)

    sentence_list = []

    subjects = sorted(os.listdir(unaligned_path))
    for subject in subjects:
        subject_path = os.path.join(unaligned_path, subject)
        sentences = sorted(os.listdir(subject_path))
        for sentence in sentences:
            sentence_path = os.path.join(subject_path, sentence)
            sentence_list.append(sentence_path)

    for sentence in tqdm(sentence_list):
        path_split = sentence.split('/')
        subject_name = path_split[-2]
        sentence_name = path_split[-1]

        save_path = os.path.join(dir_path, 'aligned_to_voca_root', 
                                 subject_name, sentence_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            continue

        f_list = gen_file_list(sentence, ext='.ply')

        mesh = Mesh(f_list)
        mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
        mesh.get_vertex_postions(mesh_vertices, no_progress=True)

        mesh.mesh_alignment(mesh_vertices, root_mesh_vertices)

        mesh_vertices = mesh.vertices_to_2d(mesh_vertices)

        file_name = 'aligned'
        mesh.export_all_meshes(mesh_vertices, save_path, file_name)
