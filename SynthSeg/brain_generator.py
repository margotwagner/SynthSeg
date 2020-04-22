# project imports
from .model_inputs import build_model_inputs
from .labels_to_image_model import labels_to_image_model

# third-party imports
from ext.lab2im import utils


class BrainGenerator:

    def __init__(self,
                 labels_dir,
                 generation_labels=None,
                 output_labels=None,
                 n_neutral_labels=None,
                 batch_size=1,
                 n_channels=1,
                 target_res=None,
                 output_shape=None,
                 output_div_by_n=None,
                 padding_margin=None,
                 flipping=True,
                 prior_distributions='uniform',
                 prior_means=None,
                 prior_stds=None,
                 use_specific_stats_for_channel=False,
                 generation_classes=None,
                 apply_linear_trans=True,
                 scaling_bounds=None,
                 rotation_bounds=None,
                 shearing_bounds=None,
                 apply_nonlin_trans=True,
                 nonlin_std=3.,
                 nonlin_shape_factor=0.0625,
                 blur_background=True,
                 data_res=None,
                 thickness=None,
                 downsample=False,
                 blur_range=1.15,
                 crop_channel_2=None,
                 apply_bias_field=True,
                 bias_field_std=0.3,
                 bias_shape_factor=0.025):
        """
        This class is wrapper around the labels_to_image_model model. It contains the GPU model that generates images
        from labels maps, and a python generator that suplies the input data for this model.
        To generate pairs of image/labels you can just call the method generate_image() on an object of this class.
        :param labels_dir: path of folder with all input label maps
        :param generation_labels: (optional) list of all possible label values in the input label maps.
        Should be organised as follows: background label first, then non-sided labels (e.g. CSF, brainstem, etc.), then
        all the structures of the same hemisphere (can be left or right), and finally all the corresponding
        contralateral structures (in the same order).
        Can be a sequence or a 1d numpy array, or the path to a 1d numpy array.
        Default is None, where the label values are directly gotten from the provided label maps.
        :param output_labels: (optional) list of all the label values to keep in the output label maps, in no
        particular order. Label values that are in generation_labels but not in output_labels are reset to zero.
        Can be a sequence or a 1d numpy array, or the path to a 1d numpy array.
        :param n_neutral_labels: (optional) number of non-sided generation labels.
        Default is total number of label values.
        :param batch_size: (optional) numbers of images to generate per mini-batch. Default is 1.
        :param n_channels: (optional) number of channels to be synthetised. Default is 1.
        :param target_res: (optional) target resolution of the generated images and corresponding label maps.
        If None, the outputs will have the same resolution as the input label maps.
        Can be a number (isotropic resolution), a sequence, a 1d numpy array, or the path to a 1d numpy array.
        :param output_shape: (optional) desired shape of the output images.
        If the atlas and target resolutions are the same, the output will be cropped to output_shape, and if the two
        resolutions are different, the output will be resized with trilinear interpolation to output_shape.
        Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.
        :param output_div_by_n: (optional) forces the output shape to be divisible by this value. It overwrites
        output_shape if necessary. Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or
        the path to a 1d numpy array..
        :param padding_margin: (optional) margin by which to pad the input labels with zeros.
        Padding is applied prior to any other operation.
        Can be an integer (same padding in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy
        array. Default is no padding.
        :param flipping: (optional) whether to introduce right/left random flipping. Default is True.
        :param prior_distributions: (optional) type of distribution from which we sample the GMM parameters.
        Can either be 'uniform', or 'normal'. Default is 'uniform'.
        :param prior_means: (optional) hyperparameters controlling the prior distributions of the GMM means. Because
        these prior distributions are uniform or normal, they require by 2 hyperparameters. Thus prior_means can be:
        1) a sequence of length 2, directly defining the two hyperparameters: [min, max] if prior_distributions is
        uniform, [mean, std] if the distribution is normal. The GMM means of are independently sampled at each
        mini_batch from the same distribution.
        2) an array of shape (2, n_labels). The mean of the Gaussian distribution associated to label k is sampled at
        each mini_batch from U(prior_means[0,k], prior_means[1,k]) if prior_distributions is uniform, and from
        N(prior_means[0,k], prior_means[1,k]) if prior_distributions is normal.
        3) an array of shape (2*n_mod, n_labels), where each block of two rows is associated to hyperparameters derived
        from different modalities. In this case, if use_specific_stats_for_channel is False, we first randomly select a
        modality from the n_mod possibilities, and we sample the GMM means like in 2).
        If use_specific_stats_for_channel is True, each block of two rows correspond to a different channel
        (n_mod=n_channels), thus we select the corresponding block to each channel rather than randomly drawing it.
        4) the path to such a numpy array.
        Default is None, which corresponds to prior_means = [25, 225].
        :param prior_stds: (optional) same as prior_means but for the standard deviations of the GMM.
        Default is None, which corresponds to prior_stds = [5, 25].
        :param use_specific_stats_for_channel: (optional) whether the i-th block of two rows in the prior arrays must be
        only used to generate the i-th channel. If True, n_mod should be equal to n_channels. Default is False.
        :param generation_classes: (optional) Indices regrouping generation labels into classes when sampling the GMM.
        Intensities of corresponding to regouped labels will thus be sampled from the same distribution. Must have the
        same length as generation_labels. Can be a sequence, a 1d numpy array, or the path to a 1d numpy array.
        Default is all labels have different classes.
        :param apply_linear_trans: (optional) whether to apply affine deformation. Default is True.
        :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
        sampled from a uniform distribution of predefined bounds. Can either be:
        1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
        (1-scaling_bounds, 1+scaling_bounds) for each dimension.
        2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
        (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
        3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
         of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
        4) the path to such a numpy array.
        If None (default), scaling_range = 0.15
        :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
        and 2, the bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
        If None (default), rotation_bounds = 15.
        :param shearing_bounds: (optional) same as scaling bounds. If None (default), shearing_bounds = 0.01.
        :param apply_nonlin_trans: (optional) whether to apply non linear elastic deformation.
        If true, a diffeomorphic deformation field is obtained by first sampling a small tensor from the normal
        distribution, resizing it to image size, and integrationg it. Default is True.
        :param nonlin_std: (optional) If apply_nonlin_trans is True, standard deviation of the normal distribution
        from which we sample the first tensor for synthesising the deformation field.
        :param nonlin_shape_factor: (optional) If apply_nonlin_trans is True, ratio between the size of the input label
        maps and the size of the sampled tensor for synthesising the deformation field.
        :param blur_background: (optional) If True, the background is blurred with the other labels, and can be reset to
        zero with a probability of 0.2. If False, the background is not blurred (we apply an edge blurring correction),
        and can be replaced by a low-intensity background with a probability of 0.5.
        :param data_res: (optional) If provided, the generated images are blurred to mimick data that would be: acquired
        at the given lower resolution, and then resampled at target_resolution. Default is None, where images are
        slightly isotropically blurred to introduce some spatial correlation between voxels.
        Can be an number (isotropic resolution), a sequence, a 1d numpy array, or the path to a 1d numpy array.
        :param thickness: (optional) if data_res is provided, we can further specify the slice thickness of the low
        resolution images to mimick.
        Can be a number (isotropic thickness), a sequence, a 1d numpy array, or the path to a 1d numpy array.
        :param downsample: (optional) whether to actually downsample the volume image to data_res. Default is False.
        :param blur_range: (optional) Randomise the standard deviation of the blurring kernels, (whether data_res is
        given or not). At each mini_batch, the standard deviation of the blurring kernels are multiplied by a c
        oefficient sampled from a uniform distribution with bounds [1/blur_range, blur_range].
        If None, no randomisation. Default is 1.15.
        :param crop_channel_2: (optional) stats for cropping second channel along the anterior-posterior axis.
        Should be a vector of length 4, with bounds of uniform distribution for cropping the front and back of the image
        (in percentage). None is no croppping.
        :param apply_bias_field: (optional) whether to apply a bias field to the final image. Default is True.
        If True, the bias field is obtained by sampling a first tensor from normal distribution, resizing it to image
        size, and rescaling the values to positive number by taking the voxel-wise exponential. Default is True.
        :param bias_field_std: (optional) If apply_nonlin_trans is True, standard deviation of the normal
        distribution from which we sample the first tensor for synthesising the bias field.
        :param bias_shape_factor: (optional) If apply_bias_field is True, ratio between the size of the input
        label maps and the size of the sampled tensor for synthesising the bias field.
        """

        # prepare data files
        if ('.nii.gz' in labels_dir) | ('.nii' in labels_dir) | ('.mgz' in labels_dir) | ('.npz' in labels_dir):
            self.labels_paths = [labels_dir]
        else:
            self.labels_paths = utils.list_images_in_folder(labels_dir)
        assert len(self.labels_paths) > 0, "Could not find any training data"

        # generation parameters
        self.labels_shape, self.aff, self.n_dims, _, self.header, self.atlas_res = \
            utils.get_volume_info(self.labels_paths[0])
        self.n_channels = n_channels
        if generation_labels is not None:
            self.generation_labels = generation_labels
        else:
            self.generation_labels = utils.get_list_labels(labels_dir=labels_dir)
        if output_labels is not None:
            self.output_labels = output_labels
        else:
            self.output_labels = self.generation_labels
        if n_neutral_labels is not None:
            self.n_neutral_labels = n_neutral_labels
        else:
            self.n_neutral_labels = generation_labels.shape[0]
        self.target_res = utils.load_array_if_path(target_res)
        # preliminary operations
        self.padding_margin = utils.load_array_if_path(padding_margin)
        self.flipping = flipping
        self.output_shape = utils.load_array_if_path(output_shape)
        self.output_div_by_n = output_div_by_n
        # GMM parameters
        self.prior_distributions = prior_distributions
        self.prior_means = utils.load_array_if_path(prior_means)
        self.prior_stds = utils.load_array_if_path(prior_stds)
        self.use_specific_stats_for_channel = use_specific_stats_for_channel
        self.generation_classes = utils.load_array_if_path(generation_classes)
        # linear transformation parameters
        self.apply_linear_trans = apply_linear_trans
        self.scaling_bounds = utils.load_array_if_path(scaling_bounds)
        self.rotation_bounds = utils.load_array_if_path(rotation_bounds)
        self.shearing_bounds = utils.load_array_if_path(shearing_bounds)
        # elastic transformation parameters
        self.apply_nonlin_trans = apply_nonlin_trans
        self.nonlin_std = nonlin_std
        self.nonlin_shape_factor = nonlin_shape_factor
        # blurring parameters
        self.blur_background = blur_background
        self.data_res = utils.load_array_if_path(data_res)
        self.thickness = utils.load_array_if_path(thickness)
        self.downsample = downsample
        self.blur_range = blur_range
        self.crop_second_channel = utils.load_array_if_path(crop_channel_2)
        # bias field parameters
        self.apply_bias_field = apply_bias_field
        self.bias_field_std = bias_field_std
        self.bias_shape_factor = bias_shape_factor

        # build transformation model
        self.labels_to_image_model, self.model_output_shape = self._build_labels_to_image_model()

        # build generator for model inputs
        self.model_inputs_generator = self._build_model_inputs_generator(batch_size)

        # build brain generator
        self.brain_generator = self._build_brain_generator()

    def _build_labels_to_image_model(self):
        # build_model
        lab_to_im_model = labels_to_image_model(labels_shape=self.labels_shape,
                                                n_channels=self.n_channels,
                                                generation_labels=self.generation_labels,
                                                output_labels=self.output_labels,
                                                n_neutral_labels=self.n_neutral_labels,
                                                atlas_res=self.atlas_res,
                                                target_res=self.target_res,
                                                output_shape=self.output_shape,
                                                output_div_by_n=self.output_div_by_n,
                                                padding_margin=self.padding_margin,
                                                flipping=self.flipping,
                                                aff=self.aff,
                                                apply_linear_trans=self.apply_linear_trans,
                                                apply_nonlin_trans=self.apply_nonlin_trans,
                                                nonlin_std=self.nonlin_std,
                                                nonlin_shape_factor=self.nonlin_shape_factor,
                                                blur_background=self.blur_background,
                                                data_res=self.data_res,
                                                thickness=self.thickness,
                                                downsample=self.downsample,
                                                blur_range=self.blur_range,
                                                crop_channel2=self.crop_second_channel,
                                                apply_bias_field=self.apply_bias_field,
                                                bias_field_std=self.bias_field_std,
                                                bias_shape_factor=self.bias_shape_factor)
        out_shape = lab_to_im_model.output[0].get_shape().as_list()[1:]
        return lab_to_im_model, out_shape

    def _build_model_inputs_generator(self, batch_size):
        # build model's inputs generator
        model_inputs_generator = build_model_inputs(path_label_maps=self.labels_paths,
                                                    n_labels=len(self.generation_labels),
                                                    batch_size=batch_size,
                                                    n_channels=self.n_channels,
                                                    generation_classes=self.generation_classes,
                                                    prior_means=self.prior_means,
                                                    prior_stds=self.prior_stds,
                                                    prior_distributions=self.prior_distributions,
                                                    use_specific_stats_for_channel=self.use_specific_stats_for_channel,
                                                    apply_linear_trans=self.apply_linear_trans,
                                                    scaling_bounds=self.scaling_bounds,
                                                    rotation_bounds=self.rotation_bounds,
                                                    shearing_bounds=self.shearing_bounds)
        return model_inputs_generator

    def _build_brain_generator(self):
        while True:
            model_inputs = next(self.model_inputs_generator)
            [image, labels] = self.labels_to_image_model.predict(model_inputs)
            yield image, labels

    def generate_brain(self):
        """call this method when an object of this class has been instantiated to generate new brains"""
        (image, labels) = next(self.brain_generator)
        return image, labels