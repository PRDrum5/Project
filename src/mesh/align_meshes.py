import os
import numpy as np
from mesh import Mesh

if __name__ == "__main__":

    # Import root mesh to align onto
    root_mesh_dir = os.path.join("/home/peter/Documents/Uni/Project/datasets/registereddata/FaceTalk_170725_00137_TA/sentence01")
    root_mesh = Mesh(root_mesh_dir, given_mesh="sentence01.000001.ply")
    root_mesh_vertices = root_mesh.get_empty_vertices(root_mesh.num_files)
    root_mesh.get_vertex_postions(root_mesh_vertices)

    # Align meshes
    for file in range(1, 2):
        sentence = 'sentence' + '%02d' %file

        dir_plys = os.path.join("/home/peter/Documents/Uni/Project/datasets/registereddata/FaceTalk_170725_00137_TA/", sentence)
        dir_path = os.path.dirname(os.path.realpath(__file__))

        mesh = Mesh(dir_plys)
        mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
        mesh.get_vertex_postions(mesh_vertices)

        mesh.mesh_alignment(mesh_vertices, root_mesh_vertices)

        mesh_vertices = mesh.vertices_to_2d(mesh_vertices)

        # Export aligned meshes
        save_path = os.path.join(dir_path, sentence)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = 'aligned'
        mesh.export_all_meshes(mesh_vertices, save_path, file_name)