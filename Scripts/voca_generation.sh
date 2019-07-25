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

    WORD_SAVE_PATH="$OUTPUT_PATH/$WORDNAME"

    for MODE in $AUDIO_PATH/$WORDNAME/*
    do
        # Get mode name
        IFS='/'
        read -ra ADDR <<< "$MODE"
        MODE_NAME=${ADDR[-1]}
        IFS=' '

        for $AUDIO in $MODE/*.wav
        do
            # Get audio clip name
            IFS='/'
            read -ra ADDR <<< "$AUDIO"
            AUDIO_NAME=${ADDR[-1]}
            IFS=' '
            echo $AUDIO_NAME
            sleep 300
        done
    done
done
