import os
import glob
import pandas as pd
import numpy as np
import ants
from nilearn.signal import clean
import nibabel as nib
from nilearn.image import resample_img
from nilearn.maskers import NiftiLabelsMasker
from scipy import ndimage
from nilearn.image import resampling
from nilearn._utils.niimg_conversions import check_niimg
from nilearn.signal import clean


class fMRI_Segmentation:
    def __init__(
        self,
        raw_path,  # Path to raw fMRI data
        sMRI,  # Path to segmented sMRI data
        saving_path,  # Directory to save segmented data
        subjects_list_path='',  # Optional path to subject list
    ):
        # User-specified parameters
        self.raw_path = raw_path
        self.sMRI = sMRI
        self.saving_path = saving_path
        self.subjects_list_path = subjects_list_path

        # Predefined parameters
        self.tot_rois = 98  # Total number of ROIs in Desikan-Killany atlas
        self.vol_to_exclude = 17  # Volumes to remove
        self.cortical_rois = 30  # From 30 onward are cortical ROIs
        self.high_pass = 0.009  # High-pass frequency
        self.low_pass = 0.08  # Low-pass frequency
        self.tr = 0.8  # Repetition time
        self.Desikan_ROIs = self._get_desikan_rois()

        # Build dataset based on whether subject list is provided
        if self.subjects_list_path == '':
            dataset = self.build_dataset()
        else:
            dataset = self.build_subset_dataset()

        # Assign raw fMRI and segmented sMRI data to instance variables
        self.raw_fmri, self.seg_smri = dataset

    def _get_desikan_rois(self):
        return [
            "left cerebral white matter", "left lateral ventricle", "left inferior lateral ventricle",
            "left cerebellum white matter", "left cerebellum cortex", "left thalamus", "left caudate",
            "left putamen", "left pallidum", "3rd ventricle", "4th ventricle", "brain-stem", "left hippocampus",
            "left amygdala", "unknown", "left accumbens area", "left ventral DC", "right cerebral white matter",
            "right lateral ventricle", "right inferior lateral ventricle", "right cerebellum white matter",
            "right cerebellum cortex", "right thalamus", "right caudate", "right putamen", "right pallidum",
            "right hippocampus", "right amygdala", "right accumbens area", "right ventral DC", "ctx-lh-bankssts",
            # Add remaining cortical ROIs here ...
        ]

    def build_subset_dataset(self):
        """
        Builds a dataset of subject-specific fMRI and segmented sMRI files based on a provided subject list.
        The subject list file must contain one subject per line.
        """
        # Read the subject list from the provided text file
        with open(self.subjects_list_path, "r") as file:
            subjects_to_include = [line.strip() for line in file]

        # Find matching fMRI files
        raw_fmri_files = self._find_matching_files(self.raw_path, subjects_to_include, file_pattern="sub-*.nii")

        # Find matching segmented sMRI files
        seg_smri_files = self._find_matching_files(self.sMRI, subjects_to_include, file_pattern="sub-*synthseg.nii")

        print(f'Dataset ready with {len(raw_fmri_files)} fMRI files and {len(seg_smri_files)} sMRI files.')

        return raw_fmri_files, seg_smri_files

    def _find_matching_files(self, base_path, subjects, file_pattern):
        """
        Helper function to find files that match specific subjects.
        """
        matching_files = []
        for filename in glob.iglob(os.path.join(base_path, "**", file_pattern), recursive=True):
            if any(subject in filename for subject in subjects):
                matching_files.append(filename)
        return matching_files

    def build_dataset(self):
        """
        Builds a dataset with all available fMRI and segmented sMRI files.
        """
        raw_fmri_files = glob.glob(os.path.join(self.raw_path, "**", "sub-*.nii"), recursive=True)
        seg_smri_files = glob.glob(os.path.join(self.sMRI, "**", "sub-*synthseg.nii"), recursive=True)
        
        print(f'Full dataset ready with {len(raw_fmri_files)} fMRI files and {len(seg_smri_files)} sMRI files.')

        return raw_fmri_files, seg_smri_files


    def get_parcellation_coordinates(self, label_img):
        # Check if input is a valid image and extract data and affine
        label_img = check_niimg(label_img)
        label_data = label_img.get_fdata()
        label_affine = label_img.affine

        # Get unique labels in the image (excluding background label 0)
        unique_labels = np.unique(label_data)[1:]

        # List to store center of mass coordinates
        coordinates_list = []

        for label in unique_labels:
            # Create a binary mask for the current label
            label_mask = label_data == label

            # Find connected components within the label
            connected_components, num_components = ndimage.label(label_mask)

            if num_components == 0:
                raise ValueError(f"No connected components found for label {label}")
            
            # If multiple components exist, focus on the largest one
            if num_components > 1:
                sizes = ndimage.sum(label_mask, connected_components, range(1, num_components + 1))
                largest_component = sizes.argmax() + 1
                largest_component_mask = connected_components == largest_component
            else:
                largest_component_mask = label_mask

            # Compute the center of mass of the largest connected component
            center_of_mass = ndimage.center_of_mass(largest_component_mask)
            
            # Transform voxel coordinates into world coordinates
            world_coords = resampling.coord_transform(center_of_mass[0], center_of_mass[1], center_of_mass[2], label_affine)
            
            # Append the world coordinates to the list
            coordinates_list.append(world_coords)

        return coordinates_list


    def band_pass_filt_fmri(self):
        raw_filenames = ["fMRI_segmented"]

        for raw in raw_filenames:
            data_to_filter = glob.glob(os.path.join(self.saving_path, "**", f"{raw}*.csv"), recursive=True)

            for data_path in data_to_filter:
                run = data_path[-10:-4]  # Assuming the last 10-4 characters indicate the run
                subject = [item for item in data_path.split("/") if item.startswith("sub-") and len(item) < 20][0]

                # Output filtered file path
                filtered_output_path = os.path.join(self.saving_path, subject, f"filt_{raw}_{subject}_{run}.csv")

                # Check if the filtered file already exists
                if not os.path.exists(filtered_output_path):
                    try:
                        # Load time series data to be filtered
                        time_series = pd.read_csv(data_path).values

                        # Apply band-pass filter using Nilearn
                        time_series_filtered = clean(
                            time_series, 
                            low_pass=self.low_pass, 
                            high_pass=self.high_pass, 
                            t_r=self.tr
                        )

                        # Create directory if it doesn't exist
                        subject_dir = os.path.join(self.saving_path, subject)
                        os.makedirs(subject_dir, exist_ok=True)

                        # Save filtered time series to CSV
                        pd.DataFrame(time_series_filtered).to_csv(filtered_output_path, index=False)
                        print(f"Filtered data saved for {subject}, run: {run}")

                    except Exception as e:
                        print(f"Error filtering data for {subject}, run: {run}. Error: {str(e)}")
                else:
                    print(f"Filtered file already exists for {subject}, run: {run}")


    def segment_fmri(self):
        error_log = []
        for i, data in enumerate(self.raw_fmri):
            run = data.split("_")[-2]
            subject = [item for item in data.split("/") if item.startswith("sub-") and len(item) < 20][0]
            
            # Check if the fMRI file is already segmented
            segmented_fmri_path = os.path.join(self.saving_path, subject, f"fMRI_segmented_{subject}_{run}.csv")
            if os.path.exists(segmented_fmri_path):
                print('exists')
                continue

            print(f"Segmenting {subject} ({i})")
            
            # Load corresponding segmented sMRI data
            sMRI_files = [s for s in self.seg_smri if subject in s]
            if len(sMRI_files) == 0:
                print(f"No sMRI file found for {subject}")
                error_log.append(f"Missing sMRI: {subject}")
                continue
            
            try:
                img_seg = nib.load(sMRI_files[0])
                fMRI_img = nib.load(data)

                # Select a volume from fMRI
                fMRI_volume = fMRI_img.slicer[:, :, :, 20]

                # Load motion confounds file (if available)
                confounds_path = data[:-8] + "motion.tsv"
                confounds = None
                if os.path.exists(confounds_path):
                    confounds = confounds_path

                # Resample sMRI segmentation to match fMRI resolution
                res_img_seg = resample_img(img_seg, target_affine=fMRI_volume.affine, target_shape=fMRI_volume.shape, interpolation="nearest")

                # Transform the original values in sMRI segmentation to a continuous range
                original_values = np.unique(res_img_seg.get_fdata())
                new_values = list(range(len(original_values)))
                value_mapping = dict(zip(original_values, new_values))
                transformed_image_data = np.vectorize(value_mapping.get)(res_img_seg.get_fdata())
                transformed_nifti_image = nib.Nifti1Image(transformed_image_data, res_img_seg.affine, dtype=np.int64)

                # Convert NIfTI images to ANTs images
                target_ants = ants.from_numpy(fMRI_volume.get_fdata(), origin=tuple(fMRI_volume.affine[:3, 3]), spacing=tuple(fMRI_volume.header.get_zooms()))
                source_label_ants = ants.from_numpy(transformed_nifti_image.get_fdata(), origin=tuple(transformed_nifti_image.affine[:3, 3]), spacing=tuple(transformed_nifti_image.header.get_zooms()))

                # Perform rigid registration to align fMRI and sMRI segmentation
                registration_result = ants.registration(fixed=target_ants, moving=source_label_ants, type_of_transform='Rigid', interpolation='None')

                # Apply the transformation to the sMRI segmentation
                transformed_label = ants.apply_transforms(fixed=target_ants, moving=source_label_ants, transformlist=registration_result['fwdtransforms'], interpolation='None')

                # Convert transformed label back to NIfTI format
                transformed_label_nifti = nib.Nifti1Image(transformed_label.numpy().astype(np.int32), affine=fMRI_volume.affine)

                # Extract ROI time series using NiftiLabelsMasker
                masker = NiftiLabelsMasker(labels_img=transformed_label_nifti, standardize='zscore_sample')

                try:
                    fMRI_seg_ts = masker.fit_transform(fMRI_img, confounds=confounds)

                    # Check for missing or extra ROIs
                    if fMRI_seg_ts.shape[1] < self.tot_rois:
                        print(f"Missing ROIs for {subject}!")
                        error_log.append(f"Missing ROIs: {subject}")
                        continue
                    elif fMRI_seg_ts.shape[1] > self.tot_rois:
                        print(f"Extra ROIs detected for {subject}")

                    # Exclude noisy initial volumes and select cortical ROIs
                    fMRI_seg_ts = fMRI_seg_ts[self.vol_to_exclude:, :]
                    # cortex_time_series = fMRI_seg_ts[:, self.cortical_rois:]

                    # Save segmented fMRI time series
                    subject_path = os.path.join(self.saving_path, subject)
                    os.makedirs(subject_path, exist_ok=True)

                    # pd.DataFrame(cortex_time_series).to_csv(f"cortex_fMRI_segmented_{subject}_{run}.csv", index=None)
                    pd.DataFrame(fMRI_seg_ts).to_csv(f"{subject_path}/fMRI_segmented_{subject}_{run}.csv", index=None)
                    print(f"Segmentation saved at {subject_path}/fMRI_segmented_{subject}_{run}.csv")

                except Exception as confound_error:
                    print(f"No confounds file or error for {subject}: {confound_error}")
                    error_log.append(f"Confound error: {subject}")
            except Exception as e:
                print(f"Error segmenting {subject}: {e}")
                error_log.append(f"Segmentation error: {subject}")
        
        return error_log
