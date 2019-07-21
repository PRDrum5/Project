#!/bin/bash

# Source file path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

LRW_VIDEO_DIR="$DIR/lipread_mp4"
LRW_AUDIO_DIR="$DIR/lrw_audio"

for DIR in $LRW_VIDEO_DIR/*
do
    # Get word name
    IFS='/'
    read -ra ADDR <<< "$DIR"
    WORDNAME=${ADDR[-1]}
    IFS=' '

    SAVE_PATH="$LRW_AUDIO_DIR/$WORDNAME"

    for MODE in $LRW_VIDEO_DIR/$WORDNAME/*
    do
        # Get mode name
        IFS='/'
        read -ra ADDR <<< "$MODE"
        MODE_NAME=${ADDR[-1]}
        IFS=' '

        for VIDEO in $MODE/*.mp4
        do
            # Get mode name
            IFS='/'
            read -ra ADDR <<< "$VIDEO"
            VIDEO_NAME=${ADDR[-1]}
            IFS=' '

            VIDEO_NAME="${VIDEO_NAME%.*}"
            VIDEO_NAME="${VIDEO_NAME##*/}"

            VIDEO_IN="$VIDEO"
            AUDIO_OUT_PATH="$SAVE_PATH/$MODE_NAME"
            mkdir -p $AUDIO_OUT_PATH
            AUDIO_OUT="$AUDIO_OUT_PATH/$VIDEO_NAME.wav"

            ffmpeg -i $VIDEO_IN -vn -acodec pcm_s16le -ar 22000 -ac 2 $AUDIO_OUT -hide_banner -loglevel panic
        done
    done

done
