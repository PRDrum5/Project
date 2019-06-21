#!/bin/bash
# This script takes wav files under the SAMPLES directory and trims them
#   into 0.1 second clips, the results are exported to OUTPUT

# Source file path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Source of original audio samples
SAMPLES="$DIR/samples"

# Trimmed output path
OUTPUT="$DIR/trimmed_samples"

# File extension
EXTENSION=".wav"

# Crop duration
DURATION=0.1

# Temp file name
TEMP_FILE="temp_file"$EXTENSION

THRESHOLD=0.00000001

# Make filepath for trimmed audio samples to live
mkdir -p $OUTPUT

for file in $SAMPLES/*.wav
do
    filename="${file%.*}"
    filename="${filename##*/}"

    # Backup file
    backup=$SAMPLES/$filename"_bkup"$EXTENSION
    cp $file $backup

    # Get duration of file
    file_length="$(soxi -D $file)"

    output_path=$OUTPUT/$filename
    mkdir -p $output_path

    cp $file $TEMP_FILE

    # Get number of slices the audio clip will be cut into.
    let "slices = $(bc <<< "$file_length / $DURATION")"

    for i in $(seq -f "%04g" 1 $slices)
    do
        outfile=$output_path/$filename_$i$EXTENSION
        sox $file $outfile trim 0 $DURATION
        sox $file $TEMP_FILE trim 0.1
        mv $TEMP_FILE $file

        # Check that the amplitude of the file isn't zero
        amplitude="$(sox $outfile -n stat 2>&1 | sed -n 's#^Maximum amplitude:[^0-9]*\([0-9.]*\)$#\1#p')"
        if (( $(echo "$amplitude < $THRESHOLD" |bc -l) )); then
            echo $outfile
            rm $outfile
            break
        fi
    done

    # Recover original from backup
    rm $file
    mv $backup $file
done

# File can be reconstructed by calling from the dir containing the sentence
#   sox *.wav out.wav
