from plyfile import PlyData
import os
import numpy as np
import time

filename = "head.ply"
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, filename)
ply_files = "/home/peter/Documents/Uni/Project/datasets/registereddata/FaceTalk_170725_00137_TA/sentence01"

def ply_verts_to_array(ply_verts):
    length = ply_verts.count
    verts = np.empty((length, 3))

    for vert in range(length):
        v = ply_verts[vert]
        (x, y, z) = (v[t] for t in ('x', 'y', 'z'))
        verts[vert] = [x, y, z]

    return verts

def list_of_plys(filepath):
    """
    Returns a list of ply files in given directory
    """
    files = []
    for file in os.listdir(filepath):
        if file.endswith("1.ply"):
            files.append(file)
    return files

verts = []

files = list_of_plys(ply_files)
for file in files:
    filepath = os.path.join(ply_files, file)

    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
        face_verts = ply_verts_to_array(plydata['vertex'])
    verts.append(face_verts)

print(len(verts))