# fMRI Segmentation Script

## Overview

This script facilitates the segmentation of functional MRI (fMRI) data using an existing sMRI segmentation obtained through SynthSeg+
The user must specify paths for raw data, segmentated data, and output locations. It allows users to input necessary paths via the terminal and handles the segmentation process while offering an option to save any encountered errors.

## Requirements

- Python 3.x
- Required libraries: 
  - NumPy
  - Pandas
  - scipy
  - Nilearn
  - Nibabel
  - ants
  - glob
  - Other libraries may be needed depending on the `fMRI_Segmentation` module implementation.

## Installation

1. Clone this repository or download the script.
2. Ensure you have Python installed along with the required libraries.
3. Modify the import statement if the `fMRI_Segmentation` module is located in a different directory.

## Usage

To run the script, execute the following command in the terminal:

```bash
python segment_fmri.py
```

### Input Prompts

The script will prompt the user to enter the following paths:

1. **Raw Path**: The directory containing the raw fMRI data.
   - Example: `/snl/abcd/raw-data/fmri/baseline/nback/`
   
2. **Segmentation Path**: The directory containing the segmentated sMRI data.
   - Example: `/nadata/cnl/abcd/data/imaging/smri/interim/synthseg/baseline/`

3. **Save Path**: The directory where the segmented fMRI data will be saved.
   - Example: `/cnl/abcd/data/imaging/fmri/nback/interim/segmented/baseline/`

4. **Subjects List Path**: The path to the text file containing a list of subjects to process.
   - Example: `/home/acamassa/ABCD/abcd-taskfMRI/notebooks/FC/nback_subjects.txt`


### Saving Errors

After segmentation, the script asks if the user wants to save any errors encountered during the process. If the user chooses to save errors, they will be prompted to provide a path to save the error log.

## Example Execution

```plaintext
Enter the raw path (e.g., /snl/abcd/raw-data/fmri/baseline/nback/): 
Enter the segmentation path (e.g., /nadata/cnl/abcd/data/imaging/smri/interim/synthseg/baseline/): 
Enter the save path (e.g., /cnl/abcd/data/imaging/fmri/nback/interim/segmented/baseline/): 
Enter the subjects list path (e.g., /home/acamassa/ABCD/abcd-taskfMRI/notebooks/FC/nback_subjects.txt): 
Do you want to save the errors? (yes/no): 
```
## Acknowledgments

- [fMRI_Segmentation](link_to_fMRI_Segmentation_repository) for providing the segmentation functionality.
- ABCD Study for providing valuable data for analysis.

---
