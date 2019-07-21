import os
import numpy as np
import argparse
from mesh import Mesh
from mesh import gen_file_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Blendshape Generation')
    parser.add_argument('--ply_dir', default='all_aligned/')
    parser.add_argument('--use_delta', default=True)
    parser.add_argument('--n_components', default='100')
    parser.add_argument('--shapes_name', default='shape_axis')

    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    n_components = int(args.n_components)

    # Load aligned meshes to create blendshapes
    dir_plys = args.ply_dir

    blendshape_axis_dir = 'blendshape_axis'
    blendshape_axis_path = os.path.join(dir_path, blendshape_axis_dir)
    if not os.path.exists(blendshape_axis_path):
        os.mkdir(blendshape_axis_path)

    ply_path = os.path.join(dir_path, dir_plys)

    ply_list = gen_file_list(ply_path, ext='.ply')

    mesh = Mesh(ply_list)
    mesh_vertices = mesh.get_empty_vertices(mesh.num_files, dtype=np.float32)
    mesh.get_vertex_postions(mesh_vertices)
    mesh_vertices = mesh.vertices_to_2d(mesh_vertices)

    if args.use_delta:
        # Use difference between each frame
        frame_deltas = mesh.create_frame_deltas(mesh_vertices)
        # vertices should have shape (samples x features) use transpose.
        shapes = mesh.inc_create_blendshapes(frame_deltas.T, 
                                             n_components=n_components)
    else:
        # Use difference from first frame
        first_frame_diff = mesh.create_frame_zero_diff(mesh_vertices)
        # vertices should have shape (samples x features) use transpose
        shapes = mesh.inc_create_blendshapes(first_frame_diff.T, 
                                             n_components=n_components)

    save_file_name = args.shapes_name + '_%03d.npy' % n_components

    file_save_path = os.path.join(blendshape_axis_path, save_file_name)
    np.save(file_save_path, shapes)
