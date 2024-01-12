#!/usr/bin/bash

## ---------------------------
##
## PURPOSE
##   This script runs the SynthSeg model on smri images of interest to segment ##   them into regions
##
## AUTHOR(S)
##   Margot Wagner (mwagner@ucsd.edu)
##
## DATE CREATED
##   2023-02-03
##
## TODO
##   - Change batch and event entry to use CLI with flags and error checking (usage statement)
##
## ---------------------------
##
## Usage Notes: 
## - Set DATA_DIR and OUTPUT_DIR paths as needed (must be same type)
##   
## ---------------------------

# set number of threads to use (tako has 2336)
N_THREADS=500
BATCH_NO=0014
EVENT="year2"

# set paths
DATA_DIR="/snl/abcd/raw-data/smri/$EVENT/batch-$BATCH_NO/"
OUTPUT_DIR="/cnl/abcd/data/imaging/smri/interim/synthseg/$EVENT/batch-$BATCH_NO/segmentations/"
SYNTHSEG_DIR="./commands/SynthSeg_predict.py"
VOL_PATH="/cnl/abcd/data/imaging/smri/interim/synthseg/$EVENT/batch-$BATCH_NO/volumes-$BATCH_NO.csv"
QC_PATH="/cnl/abcd/data/imaging/smri/interim/synthseg/$EVENT/batch-$BATCH_NO/qc-$BATCH_NO.csv"
PROBS_DIR="/cnl/abcd/data/imaging/smri/interim/synthseg/$EVENT/batch-$BATCH_NO/posteriors/"
#RESAMPLE_DIR="/cnl/abcd/data/imaging/smri/interim/synthseg/baseline/resampled/"

# activate python 3.8 environment
source "/home/mwagner/bin/anaconda3/etc/profile.d/conda.sh"
conda activate synthseg

# run synthseg on folder
# see https://github.com/BBillot/SynthSeg for additional flag options
cmd="python $SYNTHSEG_DIR --i $DATA_DIR --o $OUTPUT_DIR --parc --robust --vol $VOL_PATH --qc $QC_PATH --post $PROBS_DIR --threads $N_THREADS"

echo "$cmd" 
$cmd

# deactivate environment
conda deactivate
