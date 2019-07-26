import numpy as np
import os
import argparse
from plyfile import PlyData
from mesh.mesh import Mesh, gen_file_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gan Output Interpolation')

    parser.add_argument('--root_mesh_dir', 
                        default='mesh/root_meshes/voca_root')

    parser.add_argument('--blendshape_axis_dir', 
                        default='mesh/blendshape_axis')

    parser.add_argument('--blendshape_axis_file', 
                        default='shape_axis_lrw_ff_100.npy')

    parser.add_argument('--shape_params_dir', 
                        default='mesh/shape_params/shape_params_4/train')

    parser.add_argument('--shape_params_file', 
                        default='ABOUT_00001.npy')

    parser.add_argument('--out_path', 
                        default='interpolation')

    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Path to blendshape axis file
    blendshape_axis_dir = args.blendshape_axis_dir
    blendshape_axis_file = args.blendshape_axis_file
    blendshape_axis_path = os.path.join(dir_path,
                                        blendshape_axis_dir,
                                        blendshape_axis_file)

    # Path to blendshape parameters file
    shape_params_dir = args.shape_params_dir
    shape_params_file = args.shape_params_file
    shape_params_path = os.path.join(dir_path, 
                                     shape_params_dir, 
                                     shape_params_file)

    # Output path path
    save_dir = args.out_path
    save_path = os.path.join(dir_path, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## Load the root mesh to interpolate from
    # Instantiate root mesh class
    root_mesh_dir = args.root_mesh_dir
    root_mesh_path = os.path.join(dir_path, root_mesh_dir)
    root_mesh_list = gen_file_list(root_mesh_path, ext='.ply')
    root_mesh = Mesh(root_mesh_list)

    # Collect the vertices for the root mesh, and populate it
    root_vertices = root_mesh.get_empty_vertices(root_mesh.num_files)
    root_mesh.get_vertex_postions(root_vertices, no_progress=True)

    # Convert the vertices into 2d representation
    root_vertices = root_mesh.vertices_to_2d(root_vertices)


    ## Load the blendshape axis to interpolate along
    blendshape_axis = np.load(blendshape_axis_path)
    total_axis = blendshape_axis.shape[1]

    ## Load generated blendshape parameters
    shape_params = np.load(shape_params_path)
    params_used = shape_params.shape[0]

    # Remove blendshape axis not used.
    params_not_used = range(params_used, total_axis)
    blendshape_axis = np.delete(blendshape_axis, params_not_used, axis=1)

    ## Interpolate and export the root mesh for each frame in params
    for frame in range(shape_params.shape[1]):
        filename = 'face' + '%03d' % frame
        export_path = os.path.join(save_path, filename)
        frame_params = shape_params[:, frame].reshape(-1,1)
        interpolation = blendshape_axis @ frame_params
        interpolated_vertices = root_vertices + interpolation
        root_mesh.export_mesh(interpolated_vertices, 
                              root_mesh.mesh_connections, 
                              export_path)
