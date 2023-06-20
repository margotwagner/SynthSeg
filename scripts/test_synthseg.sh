#!/usr/bin/bash

## ---------------------------
##
## PURPOSE
##   This script runs the SynthSeg model on a test batch of data to verify a    ##   correct installation.
##
## AUTHOR(S)
##   Margot Wagner (mwagner@ucsd.edu)
##   Brandon Liu   (bliu6@ucsd.edu)
##
## DATE CREATED
##   2023-02-02
##
## ---------------------------
##
## Usage Notes: 
## - Set DATA_DIR and OUTPUT_DIR paths as needed
##   
## ---------------------------

# set i/o directories
# path to a scan to segment (or a folder)
DATA_DIR="/cnl/abcd/data/imaging/smri/raw/test_batch/sub-NDARINVCPPYNGF9/ses-baselineYear1Arm1/anat/sub-NDARINVCPPYNGF9_ses-baselineYear1Arm1_run-01_T1w.nii"

# path to output directory (must be same type as input)
OUTPUT_DIR="/cnl/abcd/data/imaging/smri/interim/synthseg/test_batch/sub-NDARINVCPPYNGF9_ses-baselineYear1Arm1_noparc_segmented.nii"

SYNTHSEG_DIR="./commands/SynthSeg_predict.py"

VOL_CSV="/cnl/abcd/data/imaging/smri/interim/synthseg/test_batch/sub-NDARINVCPPYNGF9_ses-baselineYear1Arm1_noparc_volumes.csv"

# activate python 3.8 environment
source "/home/mwagner/bin/anaconda3/etc/profile.d/conda.sh"
conda activate synthseg

# run SynthSeg 
# see https://github.com/BBillot/SynthSeg for additional flag options
python $SYNTHSEG_DIR --i $DATA_DIR --o $OUTPUT_DIR --fast --cpu --threads 8 --vol $VOL_CSV

# deactivate environment
conda deactivate