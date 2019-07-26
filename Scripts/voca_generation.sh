#!/bin/bash

# Source file path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

AUDIO_PATH="$DIR/audio/lrw_audio_subset"

OUTPUT_PATH="$DIR/lrw_subset_meshes"

# For each word in the dir of audio files
for WORD in $AUDIO_PATH/*
do
    # Get word name
    IFS='/'
    read -ra ADDR <<< "$WORD"
    WORDNAME=${ADDR[-1]}
    IFS=' '
    echo $WORDNAME

    WORD_SAVE_PATH="$OUTPUT_PATH/$WORDNAME"

    # Each word dir can contain 'train' 'test' or 'validation'
    for MODE in $AUDIO_PATH/$WORDNAME/*
    do
        # Get mode name
        IFS='/'
        read -ra ADDR <<< "$MODE"
        MODE_NAME=${ADDR[-1]}
        IFS=' '
        
        MODE_SAVE_PATH="$WORD_SAVE_PATH/$MODE_NAME"

        # for each sample of the given word
        for AUDIO in $MODE/*.wav
        do
            # Get audio clip name
            IFS='/'
            read -ra ADDR <<< "$AUDIO"
            AUDIO_NAME=${ADDR[-1]}
            IFS=' '

            AUDIO_NAME="${AUDIO_NAME%.*}"
            AUDIO_NAME="${AUDIO_NAME##*/}"

            # The path to where the meshes will be saved
            CLIP_MESH_SAVE_PATH="$MODE_SAVE_PATH/$AUDIO_NAME"

            echo $CLIP_MESH_SAVE_PATH
            echo $AUDIO
            #mkdir -p $CLIP_MESH_SAVE_PATH
            #python run_voca.py --audio_fname $AUDIO --out_path $CLIP_MESH_SAVE_PATH \
            #&& python3 recover_blendshape_params.py --recover_from_dir $CLIP_MESH_SAVE_PATH \
            #&&rm -r $CLIP_MESH_SAVE_PATH \
            #%% rm -r $AUDIO
            sleep 300
        done
    done
done
