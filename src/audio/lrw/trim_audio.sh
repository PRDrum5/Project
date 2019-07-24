#!/bin/bash

# Source file path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

LRW_AUDIO_DIR="$DIR/lrw_audio"

for DIR in $LRW_AUDIO_DIR/*
do
    # Get word name
    IFS='/'
    read -ra ADDR <<< "$DIR"
    WORDNAME=${ADDR[-1]}
    IFS=' '
    echo $WORDNAME

    for MODE in $LRW_AUDIO_DIR/$WORDNAME/*
    do
        # Get mode name
        IFS='/'
        read -ra ADDR <<< "$MODE"
        MODE_NAME=${ADDR[-1]}
        IFS=' '

        for AUDIO in $MODE/*.wav
        do
            # Get mode name
            IFS='/'
            read -ra ADDR <<< "$AUDIO"
            AUDIO_NAME=${ADDR[-1]}
            IFS=' '

            AUDIO_NAME="${AUDIO_NAME%.*}"
            AUDIO_NAME="${AUDIO_NAME##*/}"

            AUDIO_IN="$AUDIO"

            duration="$(soxi -D $AUDIO_IN)"
            one_sec=1.0
            if (( $(echo "$duration > $one_sec" |bc -l) ));
            then
                sox $AUDIO_IN "temp.wav" trim 0 -0.108
                sox "temp.wav" $AUDIO_IN trim 0.108 1.108
                rm "temp.wav"
            fi
        done
    done
done
