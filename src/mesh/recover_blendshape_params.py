import os
import numpy as np
from mesh import Mesh

if __name__ == "__main__":
    # Load mesh to interpolate
    dir_plys = 'sentence01'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mesh = Mesh(os.path.join(dir_path, dir_plys))
    mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
    mesh.get_vertex_postions(mesh_vertices)
    mesh_vertices = mesh.vertices_to_2d(mesh_vertices)
    verts = mesh_vertices[:,0]

    shapes = np.loadtxt(os.path.join(dir_path, 'shapes00.txt'), delimiter=',')
    total_shapes = shapes.shape[1]
    n_shapes = 5
    shapes = np.delete(shapes, range(n_shapes, total_shapes), axis=1)

    # Define blendshape parameters to interpolate along blendshpe axis.
    blendshape_params = np.random.rand(5)
    morphed_verts = mesh.morph_mesh(verts, shapes, blendshape_params)

    params = mesh.recover_blendshape_parameters(morphed_verts, shapes)
    print(params)
    print(blendshape_params)