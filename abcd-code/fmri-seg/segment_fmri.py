# from optimized_registration_segmentation import fMRI_Segmentation
from parallel_seg import fMRI_Segmentation # running in parallel on multiple CPUs
import numpy as np
import warnings

def main():

    warnings.filterwarnings("ignore")

    # Prompt user for input paths
    # raw_path = input("Enter the raw path (e.g., /snl/abcd/raw-data/fmri/baseline/nback/): ")
    # seg_smri = input("Enter the segmentation path (e.g., /nadata/cnl/abcd/data/imaging/smri/interim/synthseg/baseline/): ")
    # save_path = input("Enter the save path (e.g., /cnl/abcd/data/imaging/fmri/nback/interim/segmented/baseline/): ")
    # subjects_list_path = input("Enter the subjects list path (e.g., /home/acamassa/ABCD/abcd-taskfMRI/notebooks/FC/nback_subjects.txt): ")

    raw_path = '/snl/abcd/raw-data/fmri/baseline/rest/' 
    seg_smri = '/nadata/cnl/abcd/data/imaging/smri/interim/synthseg/baseline/' 
    save_path = '/cnl/abcd/data/imaging/fmri/rsfmri/interim/segmented/baseline/'
    subjects_list_path = '' 


    # Initialize and run the fMRI segmentation
    fmri_segmentation = fMRI_Segmentation(raw_path, seg_smri, save_path,subjects_list_path)
    error = fmri_segmentation.segment_fmri()
    # filter: band pass in the 0.009-0.8 Hz band 
    # fmri_seg.band_pass_filt_fmri()  # filtering in included in the segmentation for parallel

    # Ask if the user wants to save errors
    # save_errors = input("Do you want to save the errors? (yes/no): ").strip().lower()
    save_errors = 'yes'

    if save_errors == 'yes':
        error_file_path = save_path+'segmentation_errors.txt'
        # error_file_path = input("Enter the path to save the error file (e.g., /home/acamassa/ABCD/task_Segerrors.txt): ")
        np.savetxt(error_file_path, error, fmt='%s')
        print(f"Errors saved to {error_file_path}")
    else:
        print("Errors not saved.")

if __name__ == "__main__":
    main()
