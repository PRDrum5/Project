'''
Code from VOCA, this is not mine.
'''

import os
import glob
import argparse
from subprocess import call
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer


parser = argparse.ArgumentParser(description='Sequence visualization')

parser.add_argument('--sequence_path', 
                    default='interpolation/', 
                    help='Path to motion sequence')

parser.add_argument('--audio_fname', 
                    default='audio/all_samples/sentence001.wav', 
                    help='Path of speech sequence')

parser.add_argument('--out_path', 
                    default='./animation_visualization', 
                    help='Output path')

parser.add_argument('--video_name', 
                    default='video', 
                    help='name of output video')


args = parser.parse_args()
sequence_path = args.sequence_path
audio_fname = args.audio_fname
out_path = args.out_path

img_path = os.path.join(out_path, 'img')
if not os.path.exists(img_path):
    os.makedirs(img_path)

mv = MeshViewer()

sequence_fnames = sorted(glob.glob(os.path.join(sequence_path, '*.ply')))
if len(sequence_fnames) == 0:
    print('No meshes found')

# Render images
for frame_idx, mesh_fname in enumerate(sequence_fnames):
    frame_mesh = Mesh(filename=mesh_fname)
    mv.set_dynamic_meshes([frame_mesh], blocking=True)

    img_fname = os.path.join(img_path, '%05d.png' % frame_idx)
    mv.save_snapshot(img_fname)

# Encode images to video
cmd_audio = []
if os.path.exists(audio_fname):
    cmd_audio += ['-i', audio_fname]

video_name = ''.join([args.video_name, '.mp4'])

out_video_fname = os.path.join(out_path, video_name)
cmd = ['ffmpeg', '-framerate', '43', '-pattern_type', 'glob', '-i', os.path.join(img_path, '*.png')] + cmd_audio + [out_video_fname]
call(cmd)
