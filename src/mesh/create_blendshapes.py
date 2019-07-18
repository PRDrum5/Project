import os
import numpy as np
from mesh import Mesh
from mesh import gen_file_list

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Load aligned meshes to create blendshapes
    dir_plys = 'aligned/Subject01/sentence01/'
    dir_root_mesh = 'root_meshes/aligned_root_mesh'

    ply_path = os.path.join(dir_path, dir_plys)
    root_path = os.path.join(dir_path, dir_root_mesh)

    ply_list = gen_file_list(ply_path, ext='.ply')
    root_list = gen_file_list(root_path, ext='.ply')

    mesh = Mesh(ply_list)
    mesh_vertices = mesh.get_empty_vertices(mesh.num_files, dtype=np.float32)
    mesh.get_vertex_postions(mesh_vertices)
    mesh_vertices = mesh.vertices_to_2d(mesh_vertices)

    ## Use difference between each frame
    #frame_deltas = mesh.create_frame_deltas(mesh_vertices)
    ## vertices should have shape (samples x features) use transpose.
    ##shapes = mesh.inc_create_blendshapes(frame_deltas.T, n_components=100)
    ##shapes = mesh.new_create_blendshapes(frame_deltas.T, n_components=10)
    #shapes = mesh.create_blendshapes(frame_deltas, n_shapes=100)

    ## Use difference from first frame
    #first_frame_diff = mesh.create_frame_zero_diff(mesh_vertices) # Maybe this should be from root mesh
    ## vertices should have shape (samples x features) use transpose
    #shapes = mesh.inc_create_blendshapes(first_frame_diff.T, n_components=10)

    # Use difference from root mesh
    root_mesh = Mesh(root_list)
    root_vertices = root_mesh.get_empty_vertices(root_mesh.num_files, 
                                                 dtype=np.float32)
    root_mesh.get_vertex_postions(root_vertices)
    root_vertices = root_mesh.vertices_to_2d(root_vertices)

    root_frame_diff = mesh.create_given_frame_diff(mesh_vertices, root_vertices)
    # vertices should have shape (samples x features) use transpose
    shapes = mesh.inc_create_blendshapes(root_frame_diff.T, n_components=10)

    save_file_name = 'root_diff_shape_axis_10.npy'
    file_save_path = os.path.join(dir_path, save_file_name)
    np.save(file_save_path, shapes)
