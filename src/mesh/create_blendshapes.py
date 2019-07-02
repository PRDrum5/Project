import os
import numpy as np
from mesh import Mesh

if __name__ == "__main__":
    # Load aligned meshes to create blendshapes
    dir_plys = 'aligned'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mesh = Mesh(os.path.join(dir_path, dir_plys))
    mesh_vertices = mesh.get_empty_vertices(mesh.num_files)
    mesh.get_vertex_postions(mesh_vertices)
    mesh_vertices = mesh.vertices_to_2d(mesh_vertices)

    frame_deltas = mesh.create_frame_deltas(mesh_vertices)

    # vertices should have shape (features x samples)
    shapes = mesh.create_blendshapes(frame_deltas)
    print(shapes.shape)
    np.save('shapes00', shapes)
