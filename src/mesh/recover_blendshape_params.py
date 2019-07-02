import os
import numpy as np
from mesh import Mesh
from tqdm import tqdm

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    shapes = np.load(os.path.join(dir_path, 'shapes00.npy'))
    total_shapes = shapes.shape[1]
    n_shapes = total_shapes
    shapes = np.delete(shapes, range(n_shapes, total_shapes), axis=1)


    for file in tqdm(range(1, 41)):
        sentence = 'sentence' + '%02d' %file
        save_file = 'shape_parms_' + sentence
        dir_plys = 'aligned'

        mesh = Mesh(os.path.join(dir_path, dir_plys, sentence))
        mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
        mesh.get_vertex_postions(mesh_vertices)
        mesh_vertices = mesh.vertices_to_2d(mesh_vertices)
        verts = mesh_vertices[:,0]

        ## Define blendshape parameters to interpolate along blendshpe axis.
        #blendshape_params = np.random.rand(5)
        #morphed_verts = mesh.morph_mesh(verts, shapes, blendshape_params)

        params = np.empty((n_shapes, mesh.num_files))
        for v in range(mesh.num_files):
            params[:,v] = mesh.recover_blendshape_parameters(
                mesh_vertices[:,v], shapes)
    
        np.save(save_file, params)