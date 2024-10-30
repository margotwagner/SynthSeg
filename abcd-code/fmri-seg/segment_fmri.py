from fMRI_Segmentation import fMRI_Segmentation
import numpy as np
import warnings

def main():
    warnings.filterwarnings("ignore")


    # Prompt user for input paths
    raw_path = input("Enter the raw path (e.g., /snl/abcd/raw-data/fmri/baseline/nback/): ")
    seg_smri = input("Enter the segmentation path (e.g., /nadata/cnl/abcd/data/imaging/smri/interim/synthseg/baseline/): ")
    save_path = input("Enter the save path (e.g., /cnl/abcd/data/imaging/fmri/nback/interim/segmented/baseline/): ")
    subjects_list_path = input("Enter the subjects list path (e.g., /home/acamassa/ABCD/abcd-taskfMRI/notebooks/FC/nback_subjects.txt): ")

    # raw_path = '/snl/abcd/raw-data/fmri/baseline/rest/' 
    # seg_smri = '/nadata/cnl/abcd/data/imaging/smri/interim/synthseg/baseline/' 
    # save_path = '/cnl/abcd/test/new_pipe/' 
    # subjects_list_path = '/home/acamassa/ABCD/test.txt' 


    # Initialize and run the fMRI segmentation
    fmri_seg = fMRI_Segmentation(raw_path, seg_smri, save_path, subjects_list_path)
    error = fmri_seg.segment_fmri()
    # filter: band pass in the 0.009-0.8 Hz band 
    fmri_seg.band_pass_filt_fmri()  

    # Ask if the user wants to save errors
    save_errors = input("Do you want to save the errors? (yes/no): ").strip().lower()

    if save_errors == 'yes':
        error_file_path = input("Enter the path to save the error file (e.g., /home/acamassa/ABCD/task_Segerrors.txt): ")
        np.savetxt(error_file_path, error, fmt='%s')
        print(f"Errors saved to {error_file_path}")
    else:
        print("Errors not saved.")

if __name__ == "__main__":
    main()
