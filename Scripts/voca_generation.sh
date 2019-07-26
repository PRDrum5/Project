#!/bin/bash

# Source file path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

AUDIO_PATH="$DIR/audio/lrw_audio_subset"

OUTPUT_PATH="$DIR/lrw_subset_meshes"

for WORD in $AUDIO_PATH/*
do
    # Get word name
    IFS='/'
    read -ra ADDR <<< "$WORD"
    WORDNAME=${ADDR[-1]}
    IFS=' '
    echo $WORDNAME

    WORD_SAVE_PATH="$OUTPUT_PATH/$WORDNAME"

    for MODE in $AUDIO_PATH/$WORDNAME/*
    do
        # Get mode name
        IFS='/'
        read -ra ADDR <<< "$MODE"
        MODE_NAME=${ADDR[-1]}
        IFS=' '
        
        MODE_SAVE_PATH="$WORD_SAVE_PATH/$MODE_NAME"

        for AUDIO in $MODE/*.wav
        do
            # Get audio clip name
            IFS='/'
            read -ra ADDR <<< "$AUDIO"
            AUDIO_NAME=${ADDR[-1]}
            IFS=' '

            AUDIO_NAME="${AUDIO_NAME%.*}"
            AUDIO_NAME="${AUDIO_NAME##*/}"

            CLIP_MESH_SAVE_PATH="$MODE_SAVE_PATH/$AUDIO_NAME"

            if test -d $CLIP_MESH_SAVE_PATH;
            then
                :
            else
                mkdir -p $CLIP_MESH_SAVE_PATH
                echo $CLIP_MESH_SAVE_PATH
                sleep 300
                #python run_voca.py --audio_fname $AUDIO --out_path $CLIP_MESH_SAVE_PATH

                #python3 recover_blendshape_params.py --recover_from_dir $CLIP_MESH_SAVE_PATH
            fi
        done
    done
done
