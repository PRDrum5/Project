#!/bin/bash

PARAMS_DIR='mesh/shape_params/shape_params_10'

while getopts dp: option
do
case "${option}"
in
d) PARAMS_DIR=${OPTARG};;
p) PARAMS_FILE=${OPTARG};;
esac
done

source ~/anaconda3/etc/profile.d/conda.sh
conda activate MSc

ROOT_MESH='mesh/root_meshes/Subject01_aligned/'
BLENDSHAPE_AXIS_DIR='mesh/blendshape_axis/'
BLENDSHAPE_AXIS_FILE='shape_axis_100.npy' 
SEQ_PATH='interpolation' 
WAV_PATH='audio/all_samples/sentence001.wav'

python interpolate_gan_output.py --root_mesh_dir $ROOT_MESH --blendshape_axis_dir $BLENDSHAPE_AXIS_DIR --blendshape_axis_file $BLENDSHAPE_AXIS_FILE --shape_params_dir $PARAMS_DIR --shape_params_file $PARAMS_FILE --out_path $SEQ_PATH

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
#rm -r $SEQ_PATH