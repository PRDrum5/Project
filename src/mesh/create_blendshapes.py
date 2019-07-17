import os
import numpy as np
from mesh import Mesh
from mesh import gen_file_list

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Load aligned meshes to create blendshapes
    dir_plys = 'all_aligned'
    ply_path = os.path.join(dir_path, dir_plys)

    ply_list = gen_file_list(ply_path, ext='.ply')

    mesh = Mesh(ply_list)
    mesh_vertices = mesh.get_empty_vertices(mesh.num_files, dtype=np.float32)
    mesh.get_vertex_postions(mesh_vertices)
    mesh_vertices = mesh.vertices_to_2d(mesh_vertices)

    frame_deltas = mesh.create_frame_deltas(mesh_vertices)

    # vertices should have shape (features x samples)
    shapes = mesh.create_blendshapes(frame_deltas)

    save_file_name = 'blendshape_axis_voca'
    file_save_path = os.path.join(dir_path, save_file_name)
    np.save(file_save_path, shapes)
