# fMRI segmentation and pre processing
# inlcluding:
# 98 rois (Desikan-Killany atlas)
# removal of noisy initial volumes
# z-score
# band pass filter 0.009-0.8 Hz

# Any import statements go here, above the class declaration
import numpy as np
import glob
import os
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import mannwhitneyu
from nilearn import datasets, plotting
from nilearn import datasets, plotting
from nilearn._utils import check_niimg
from nilearn.maskers import NiftiLabelsMasker
import numpy as np
import nilearn
from nilearn.image import mean_img, index_img, resample_to_img, get_data
from nilearn._utils.ndimage import largest_connected_component
from scipy import ndimage
from nilearn.image import resampling
import nibabel as nib
import ants
from nilearn import plotting as nplot
from nilearn.plotting import plot_anat, plot_img, plot_stat_map

# this is where we declare our class
class fMRI_Segmentation:
    # this function initializes the class, so here we initialize the dataset. Every class has an initialization function
    # this is similar to the initialization you did in the notebook, so ill maintain the variable names for now

    # if you want the user to be able to specify something, put it here. Same like a function.
    def __init__(
        self,
        raw_path,  # path of raw fmri data to segment
        sMRI,  # path of segmented smri
        saving_path,  # path where save the segmented data
        subjects_list_path
    ):
        # if you want the variable to be automatically specified, only put it down here

        # for the variables you allow users to specify, you still have to tell
        # the class what they are, so it looks like this
        self.raw_path = raw_path
        self.sMRI = sMRI
        self.subjects_list_path=subjects_list_path
        self.saving_path=saving_path

        # these are automatically specified
        self.tot_rois = 98  # tot number of rois in Desikan-Killany atlas
        self.vol_to_exclude = 17  # volumes to remove
        self.cortical_rois = 30  # from 30 on are the cortical ROI
        self.hig_pass = 0.009  # high pass frequency
        self.low_pass = 0.08  # low pass frequency
        self.tr = 0.8  # time resolution
        self.saving_path = saving_path
        self.Desikan_ROIs = [
            "left cerebral white matter",
            "left lateral ventricle",
            "left inferior lateral ventricle",
            "left cerebellum white matter",
            "left cerebellum cortex",
            "left thalamus",
            "left caudate",
            "left putamen",
            "left pallidum",
            "3rd ventricle",
            "4th ventricle",
            "brain-stem",
            "left hippocampus",
            "left amygdala",
            "unknown",
            "left accumbens area",
            "left ventral DC",
            "right cerebral white matter",
            "right lateral ventricle",
            "right inferior lateral ventricle",
            "right cerebellum white matter",
            "right cerebellum cortex",
            "right thalamus",
            "right caudate",
            "right putamen",
            "right pallidum",
            "right hippocampus",
            "right amygdala",
            "right accumbens area",
            "right ventral DC",
            "ctx-lh-bankssts",
            "ctx-lh-caudalanteriorcingulate",
            "ctx-lh-caudalmiddlefrontal",
            "ctx-lh-cuneus",
            "ctx-lh-entorhinal",
            "ctx-lh-fusiform",
            "ctx-lh-inferiorparietal",
            "ctx-lh-inferiortemporal",
            "ctx-lh-isthmuscingulate",
            "ctx-lh-lateraloccipital",
            "ctx-lh-lateralorbitofrontal",
            "ctx-lh-lingual",
            "ctx-lh-medialorbitofrontal",
            "ctx-lh-middletemporal",
            "ctx-lh-parahippocampal",
            "ctx-lh-paracentral",
            "ctx-lh-parsopercularis",
            "ctx-lh-parsorbitalis",
            "ctx-lh-parstriangularis",
            "ctx-lh-pericalcarine",
            "ctx-lh-postcentral",
            "ctx-lh-posteriorcingulate",
            "ctx-lh-precentral",
            "ctx-lh-precuneus",
            "ctx-lh-rostralanteriorcingulate",
            "ctx-lh-rostralmiddlefrontal",
            "ctx-lh-superiorfrontal",
            "ctx-lh-superiorparietal",
            "ctx-lh-superiortemporal",
            "ctx-lh-supramarginal",
            "ctx-lh-frontalpole",
            "ctx-lh-temporalpole",
            "ctx-lh-transversetemporal",
            "ctx-lh-insula",
            "ctx-rh-bankssts",
            "ctx-rh-caudalanteriorcingulate",
            "ctx-rh-caudalmiddlefrontal",
            "ctx-rh-cuneus",
            "ctx-rh-entorhinal",
            "ctx-rh-fusiform",
            "ctx-rh-inferiorparietal",
            "ctx-rh-inferiortemporal",
            "ctx-rh-isthmuscingulate",
            "ctx-rh-lateraloccipital",
            "ctx-rh-lateralorbitofrontal",
            "ctx-rh-lingual",
            "ctx-rh-medialorbitofrontal",
            "ctx-rh-middletemporal",
            "ctx-rh-parahippocampal",
            "ctx-rh-paracentral",
            "ctx-rh-parsopercularis",
            "ctx-rh-parsorbitalis",
            "ctx-rh-parstriangularis",
            "ctx-rh-pericalcarine",
            "ctx-rh-postcentral",
            "ctx-rh-posteriorcingulate",
            "ctx-rh-precentral",
            "ctx-rh-precuneus",
            "ctx-rh-rostralanteriorcingulate",
            "ctx-rh-rostralmiddlefrontal",
            "ctx-rh-superiorfrontal",
            "ctx-rh-superiorparietal",
            "ctx-rh-superiortemporal",
            "ctx-rh-supramarginal",
            "ctx-rh-frontalpole",
            "ctx-rh-temporalpole",
            "ctx-rh-transversetemporal",
            "ctx-rh-insula",
        ]

        # call the function that builds the dataset
        # we can also specify it with a function
        # call the function that builds the dataset
        dataset = self.build_dataset()
        #dataset=self.build_subset_dataset(self.subjects_list_path)
        # assign the output to the variables
        (self.raw_fmri, self.seg_smri,) = dataset

        # Segmented data list
    def build_subset_dataset(self, subjects_list_path):
        # subjects_list must be a .txt file
        # Read strings to match from a text file
        with open(self.subjects_list_path, "r") as file:
            strings_to_match = [line.strip() for line in file]

        # Get a list of all files and subdirectories in the directory
        raw_fmri = []
        for filename in glob.iglob(self.raw_path + "**/sub-*.nii", recursive=True):
            raw_fmri.append(filename)

        # Filter files based on both '.nii' extension and strings in filename
        raw_fmri_subset = []
        for file_name in raw_fmri:
            for match_string in strings_to_match:
                if match_string in file_name:
                    raw_fmri_subset.append(file_name)
                    break  # No need to check other strings for this file
                        
        # Get a list of all files and subdirectories in the directory
        seg_mri = []
        for filename in glob.iglob(self.sMRI + "**/sub-*synthseg.nii", recursive=True):
            seg_mri.append(filename)

        # Filter files based on both '.nii' extension and strings in filename
        seg_mri_subset = []
        for file_name in seg_mri:
            for match_string in strings_to_match:
                if match_string in file_name:
                    seg_mri_subset.append(file_name)
                    break  # No need to check other strings for this file
        print('Dataset ready!')
                        
        return(
            raw_fmri_subset,
            seg_mri_subset,
        )
    
    
    def build_dataset(self):

        raw_fmri = []
        for filename in glob.iglob(self.raw_path + "**/sub-*.nii", recursive=True):
            raw_fmri.append(filename)
        seg_smri = []
        for filename in glob.iglob(self.sMRI + "**/sub-*synthseg.nii", recursive=True):
            seg_smri.append(filename)

        return (
            raw_fmri,
            seg_smri,
        )

    def get_parcellation_coordinates(self, label_img):
        # grab data and affine
        label_img = check_niimg(label_img)
        label_data = label_img.get_fdata()
        label_affine = label_img.affine

        # grab number of unique values in 3d image
        label_unique = np.unique(label_data)[1:]

        # grab center of mass from parcellations and dump into coords list
        coord_list = []
        for i, cur_label in enumerate(label_unique):
            cur_img = label_data == cur_label

            # take the largest connected component
            volume = np.asarray(cur_img)
            labels, label_nb = ndimage.label(volume)
            if not label_nb:
                raise ValueError("No non-zero values: no connected components")
            if label_nb == 1:
                volume = volume.astype(np.bool)

            # get parcellation center of mass
            center_of_mass = ndimage.center_of_mass(volume)
            world_coords = resampling.coord_transform(
                center_of_mass[0], center_of_mass[1], center_of_mass[2], label_affine
            )

            coord_list.append((world_coords[0], world_coords[1], world_coords[2]))

        return coord_list

    def band_pass_filt_fmri(self):
        i = 0

        raw_filename = ["cortex_fMRI_segmented", "fMRI_segmented"]
        for raw in raw_filename:

            data_to_filt = []
            for filename in glob.iglob(
                self.saving_path + "**/" + raw + "*.csv", recursive=True
            ):
                data_to_filt.append(filename)

            for data in data_to_filt:

                run = data[-10:-4]
                subject = [
                    item
                    for item in data.split("/")
                    if item.startswith("sub-") and len(item) < 20
                ][0]
                
#                 if not os.path.exists(
#                     self.saving_path
#                     + subject
#                     + "/filt_"
#                     + raw
#                     + "_"
#                     + subject
#                     + run
#                     + ".csv"
#                 ):
                print("Filtering")
                print(subject, i)
                # load time series data to filter
                ts = np.asarray(pd.read_csv(data))
                # filter using nilearn function
                ts_filt = nilearn.signal.clean(
                    ts, low_pass=self.low_pass, high_pass=self.hig_pass, t_r=self.tr
                )
                # save filtered time series
                os.chdir(self.saving_path + subject)
                ts = pd.DataFrame(ts_filt)
                ts.to_csv("filt_" + raw + "_" + subject + run + ".csv", index=None)
                i = i + 1
#                 else:
#                     print("File already exists")
#                     i = i + 1

    def segment_fmri(self):
        i = 0

        missing_rois = []
        no_confounds = []
        seg_error = []
        done = []

        for data in self.raw_fmri:
            run = data.split("_")[-2]
            subject = [
                item
                for item in data.split("/")
                if item.startswith("sub-") and len(item) < 20
            ][0]
            # check if file is already segmented
            if not os.path.exists(
                self.saving_path
                + subject
                + "/cortex_fMRI_segmented_"
                + subject
                + run
                + ".csv"
                ):

                print(subject, i)
                # try:
                #   Loading segmented sMRI data
                print(data)
                sMRI = [s for s in self.seg_smri if subject in s]
                if len(sMRI)>0:
                    img_seg = nib.load(sMRI[0])

                    # loading fmri and taking one volume
                    fMRI_img=nib.load(data)
                    try:
                        im1=fMRI_img.slicer[:,:,:,17]

                        #   Loading corresponding confound
                        confounds = data[:-8] + "motion.tsv"

                        # resampling mri to match fmri resolution
                        from nilearn.image import resample_img
                        res_img_seg = resample_img(
                        img_seg, target_affine=im1.affine, target_shape=im1.shape, interpolation="nearest"
                        )
                        # Transform the original values to continuous values
                        original_values = np.unique(res_img_seg.get_fdata())
                        new_values = list(range(len(original_values)))  # [0, 1, 2, ...]

                        # Create a mapping dictionary from original values to new values
                        value_mapping = {original_value: new_value for original_value, new_value in zip(original_values, new_values)}

                        # Apply the value mapping to the image data
                        image_data = res_img_seg.get_fdata()
                        transformed_image_data = np.vectorize(value_mapping.get)(image_data)
                        transformed_nifti_image = nib.Nifti1Image(transformed_image_data, res_img_seg.affine, dtype=np.int64)


                        # Convert the 3D target slice to ANTs image format
                        target_ants = ants.from_numpy(im1.get_fdata(), origin=tuple(im1.affine[:3, 3]), spacing=tuple(im1.header.get_zooms()))


                        # Convert the source label NIfTI image to ANTs image format
                        source_label_ants = ants.from_numpy(transformed_nifti_image.get_fdata(), origin=tuple(transformed_nifti_image.affine[:3, 3]), spacing=tuple(transformed_nifti_image.header.get_zooms()))

                        # Perform registration no interpolation to maintain labels
                        registration_result = ants.registration(fixed=target_ants, moving=source_label_ants,
                                                                type_of_transform='Rigid',aff_metric='MeanSquares',
                                                                syn_metric='MeanSquares', syn_sampling=0,total_sigma=0,
                                                                reg_iterations=(100, 50, 20),flow_sigma=0,
                                                                interpolation='None')

                        # Apply the transformation parameters to the source label image
                        transformed_label = ants.apply_transforms(fixed=target_ants,
                                                                    transformlist=registration_result['fwdtransforms'],
                                                                    moving=source_label_ants,
                                                                    interpolation='None')

                        # Convert the transformed ANTs image data to a NIfTI image
                        transformed_label_nifti = nib.Nifti1Image(transformed_label.numpy().astype(np.int32), affine=im1.affine)


                        # Masking ROIs

                        masker = NiftiLabelsMasker(labels_img=transformed_label_nifti, standardize='zscore_sample')
                        try:

                            fMRI_seg_ts = masker.fit_transform(
                                fMRI_img, confounds=confounds
                            )
                            if np.shape(fMRI_seg_ts)[1] < self.tot_rois:
                                print("missing ROIs!")
                                missing_rois.append(data)
                            elif np.shape(fMRI_seg_ts)[1] > self.tot_rois:
                                print('more ROIs than expected')
                            else:
                                fMRI_seg_ts = fMRI_seg_ts[
                                    self.vol_to_exclude :, :
                                ]  # remove noisy initial volumes
                                cortex_time_series = fMRI_seg_ts[
                                    :, self.cortical_rois :
                                ]  # select only cortical rois

                                #       Saving segmented fmri
                                os.chdir(self.saving_path)
                                isExist = os.path.exists(subject)
                                if not isExist:
                                    # Create subject directory if it does not exist
                                    os.makedirs(subject)
                                os.chdir(self.saving_path + subject)



                                #       Save cortical ROIs time series
                                ts = pd.DataFrame(cortex_time_series)
                                ts.to_csv(
                                    "cortex_fMRI_segmented_" + subject + run + ".csv",
                                    index=None,
                                )
                                ts_all = pd.DataFrame(fMRI_seg_ts)
                                done.append(
                                    self.saving_path
                                    + subject
                                    + "cortex_fMRI_segmented_"
                                    + subject
                                    + run
                                    + ".csv"
                                )
                                #       Save all time series (coertical and subcortical)
                                ts_all.to_csv(
                                    "fMRI_segmented_" + subject + run + ".csv", index=None
                                )

                        except:
                            print("no confounds file")
                            no_confounds.append(data)
                    except:
                        print("fmri too short")
                else:
                    print('no smri file')
                    # except:
                    #     seg_error.append(data)
                    #     print("File may be corrupted")
            else:
                done.append(data)
                print("File already exists")

            i = i + 1

    #         return (
    #             missing_rois,
    #             seg_error,
    #             no_confounds,
    #             done,
    #         )

    def run_segmentation_report(self, sub_to_check, DDC_path):
        # sub_to_check: list of IDs of the subjects to check
        # DDC_path: path to the DDC matrices

        df = pd.DataFrame(
            columns=[
                "subject_id",
                "raw_data",
                "segmented_fmri",
                "filtered",
                "DDC",
                "confund",
            ]
        )
        all_raw_data = []
        ddc_all = []
        segmented_all = []

        for i in range(len(sub_to_check)):
            sub = "sub-" + sub_to_check[i]

            # raw data check
            os.chdir(self.raw_path)
            if not os.path.exists(sub):
                seg = 0
                filt = 0
                ddc = 0
                f = 0
                conf = 0
            elif os.path.exists(sub):
                os.chdir(self.raw_path)
                os.chdir(sub)
                raw_files = glob.glob("**/sub-*.nii", recursive=True)

                # confounding files check
                for f in raw_files:
                    run = f.split("_")[-2]
                    all_raw_data.append(f)
                    os.chdir(self.raw_path)
                    os.chdir(sub)

                    if os.path.exists(f[:-8] + "motion.tsv"):
                        conf = f[:-8] + "motion.tsv"
                    else:
                        conf = 0
                        seg = 0
                        filt = 0
                        ddc = 0

                    # segmented data check
                    os.chdir(self.saving_path)
                    if os.path.exists(sub):
                        os.chdir(sub)
                        if os.path.exists(
                            "cortex_fMRI_segmented_" + sub + run + ".csv"
                        ):
                            seg = "cortex_fMRI_segmented_" + sub + run + ".csv"
                            segmented_all.append(seg)

                            if os.path.exists(
                                "filt_cortex_fMRI_segmented_" + sub + run + ".csv"
                            ):
                                filt = (
                                    "filt_cortex_fMRI_segmented_" + sub + run + ".csv"
                                )
                            else:
                                filt = 0

                        else:
                            seg = 0
                            filt = 0
                            ddc = 0
                    else:
                        seg = 0
                        filt = 0
                        ddc = 0

                    # DDC check
                    os.chdir(DDC_path)
                    if os.path.exists(sub):
                        os.chdir(sub + "/single_sessions")
                        if os.path.exists("subc_DDC_" + run + ".csv"):
                            ddc_all.append(
                                [
                                    DDC_path
                                    + sub
                                    + "/single_sessions/"
                                    + "Reg_DDC2H_"
                                    + run
                                    + ".csv"
                                ]
                            )
                            ddc = (
                                sub + "/single_sessions/" + "subc_DDC_" + run + ".csv"
                            )

                        else:
                            ddc = 0

                    new_raw = {
                        "subject_id": sub,
                        "raw_data": f,
                        "segmented_fmri": seg,
                        "filtered": filt,
                        "DDC": ddc,
                        "confund": conf,
                    }
                    df = pd.concat([df, pd.DataFrame([new_raw])], ignore_index=True)
                    #df = df.append(new_raw, ignore_index=True)
        return df


################################################################
