#!/bin/bash

while getopts p: option
do
case "${option}"
in
p) PARAMS=${OPTARG};;
esac
done

source ~/anaconda3/etc/profile.d/conda.sh
conda activate MSc

ROOT_MESH='model/data/root_mesh'
BLENDSHAPE_AXIS='model/data/blendshape_axis/shape_axis.npy'
SEQ_PATH='gan_visualisation'
WAV_PATH='model/data/sentence01_1sec.wav'

python interpolate_gan_output.py --root_mesh $ROOT_MESH --blendshape_axis $BLENDSHAPE_AXIS --params $PARAMS --out_path $SEQ_PATH

function join_by { local IFS="$1"; shift; echo "$*"; }

# Get outpath for video
IFS='/'
read -ra ADDR <<< "$PARAMS"
unset 'ADDR[${#ADDR[@]}-1]'
IFS=' '
SAVE_PATH=`join_by / "${ADDR[@]}"`

# Get sample filename
IFS='/'
read -ra ADDR <<< "$PARAMS"
filename=${ADDR[-1]}
IFS='.'
read -ra ADDR <<< "$filename"
filename=${ADDR[0]}
IFS=' '

conda activate voca
python visualize_sequence.py --sequence_path $SEQ_PATH --audio_fname $WAV_PATH --out_path $SAVE_PATH --video_name $filename

# Clean up
rm -r $SAVE_PATH/img
rm -r $SEQ_PATH