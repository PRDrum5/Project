#!/bin/bash

# Source file path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

LRW_VIDEO_DIR="$DIR/lipread_mp4"
LRW_AUDIO_DIR="$DIR/lrw_video"

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
            echo $VIDEO_NAME

            VIDEO_NAME="${VIDEO_NAME%.*}"
            VIDEO_NAME="${VIDEO_NAME##*/}"

            VIDEO_IN="$VIDEO"
            FRAMES_OUT_PATH="$SAVE_PATH/$MODE_NAME/$VIDEO_NAME"
            mkdir -p $FRAMES_OUT_PATH

            if test -f $FRAMES_OUT_PATH;
            then
                :
            else
                ffmpeg -i $VIDEO_IN $FRAMES_OUT_PATH/frame%02d.png -hide_banner -loglevel panic
                rm $FRAMES_OUT_PATH/frame01.png
                rm $FRAMES_OUT_PATH/frame02.png
                rm $FRAMES_OUT_PATH/frame27.png
                rm $FRAMES_OUT_PATH/frame28.png
                rm $FRAMES_OUT_PATH/frame29.png
            fi
        done
    done
done
