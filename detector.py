import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import scipy
from tqdm import tqdm
import torch
import utils.utils
from utils.utils import (
    apply_binomial_pert_dataset,
    extract_subset_features,
    fast_gradient_sign_method,
    get_shapley_values,
    get_Drebbin_dataset,
    get_discrete_derivative_inputs,
    get_flipped_samples_indices,
    get_important_features,
    get_jac,
    get_discrete_derivatives,
    get_model_name,
    get_scaled_model_output,
    generate_predictions_for_verification,
    identify_adversarial_examples,
    save_adversarial_examples_binarry_classifier,
    save_dictionary_to_file,
    scale_probability,
    verify_binary_classifier,
)
from utils.abstract import AbstractDetector
from utils.drebinnn import DrebinNN
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import (
    create_layer_map,
    load_model,
    load_models_dirpath,
    build_random_forest_classifier
)
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)


class Detector(AbstractDetector):
    def __init__(
        self,
        metaparameter_filepath,
        learned_parameters_dirpath,
        reference_model_dirpath
    ):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters 
            directory. scale_parameters_filepath: str - File path to the scale
            parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.reference_model_dirpath = reference_model_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, 
                                   "model.bin")
        self.models_padding_dict_filepath = join(
            self.learned_parameters_dirpath,
            "models_padding_dict.bin"
        )
        self.model_layer_map_filepath = join(
            self.learned_parameters_dirpath, 
            "model_layer_map.bin"
        )
        self.layer_transform_filepath = join(
            self.learned_parameters_dirpath, 
            "layer_transform.bin"
        )
        self.infer_path_adv_examples = metaparameters['infer_path_adv_examples']
        self.infer_path_drebbin_x_train = metaparameters['infer_path_drebbin_x_train']
        self.infer_path_drebbin_x_test = metaparameters['infer_path_drebbin_x_test']
        self.infer_path_drebbin_y_train = metaparameters['infer_path_drebbin_y_train']
        self.infer_path_drebbin_y_test = metaparameters['infer_path_drebbin_y_test']
        self.infer_calc_drebbin_adv = metaparameters['infer_calc_drebbin_adv']
        self.infer_grad_magnitude = metaparameters['infer_grad_magnitude']
        self.infer_save_adv_examples = metaparameters['infer_save_adv_examples']
        self.infer_stat_output_file = metaparameters['infer_stat_output_file']
        self.infer_aug_dataset_factor = metaparameters["infer_aug_dataset_factor"]
        self.infer_aug_bin_prob = metaparameters["infer_aug_bin_prob"]
        self.infer_generate_statistics = metaparameters["infer_generate_statistics"]
        self.infer_load_drebbin = metaparameters["infer_load_drebbin"]
        self.infer_platform = metaparameters["infer_platform"]
        self.feature_importance = metaparameters["infer_feature_importance"]
        self.random_noise_augmentation = metaparameters["infer_random_noise_augmentation"]
        self.no_features_least = metaparameters["infer_no_features_least"]
        self.no_features_most = metaparameters["infer_no_features_most"]
        self.infer_extra_data_augmentation = metaparameters["infer_extra_data_augmentation"]
        self.infer_path_poisoned_examples = metaparameters["infer_path_poisoned_examples"]
        self.train_random_forest_feature_importance = metaparameters["train_random_forest_feature_importance"]
        self.infer_feature_importance_path = metaparameters["infer_feature_importance_path"]
        self.infer_feature_extraction_method = metaparameters["infer_feature_extraction_method"]
        self.infer_proximity_aggregation_method = metaparameters["infer_proximity_aggregation_method"]
        if self.infer_platform == 'local':
            self.reference_model_dirpath = self.reference_model_dirpath[2:]
        self.infer_adv_examples_file_names = [
            metaparameters["infer_adv_ex_file_class01_pc0"],
            metaparameters["infer_adv_ex_file_class10_pc0"],
            metaparameters["infer_adv_ex_file_class01_pc1"],
            metaparameters["infer_adv_ex_file_class10_pc1"]
        ]

        self.infer_filename_poisoned_examples = metaparameters["infer_filename_poisoned_examples"]

    def write_metaparameters(self):
        metaparameters = {
            "infer_cyber_model_skew": self.model_skew["__all__"],
            "train_input_features": self.input_features,
            "train_weight_table_random_state":
                self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler":
                self.weight_table_params["scaler"],
            "train_random_forest_regressor_param_n_estimators":
                self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion":
                self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth":
                self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split":
                self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf":
                self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf":
                self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features":
                self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease":
                self.random_forest_kwargs["min_impurity_decrease"],
        }

        with open(join(self.learned_parameters_dirpath,
                       basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the 
        parameters from the metaparameter file, performing a 
        grid search type approach to optimize these parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from 
        the metaparameters JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = [
            join(models_dirpath, model)
            for model in listdir(models_dirpath)
        ]
        model_path_list = sorted(model_path_list)
        logging.info(f"Loading {len(model_path_list)} models...")
        (model_repr_dict, 
         model_ground_truth_dict) = load_models_dirpath(model_path_list)

        models_padding_dict = create_models_padding(model_repr_dict)
        with open(self.models_padding_dict_filepath, "wb") as fp:
            pickle.dump(models_padding_dict, fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(
                    model_repr,
                    model_class,
                    models_padding_dict
                )

        check_models_consistency(model_repr_dict)

        # Build model layer map to know how to flatten
        logging.info("Generating model layer map...")
        model_layer_map = create_layer_map(model_repr_dict)
        with open(self.model_layer_map_filepath, "wb") as fp:
            pickle.dump(model_layer_map, fp)
        logging.info("Generated model layer map. Flattenning models...")

        # Flatten models
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        layer_transform = fit_feature_reduction_algorithm(
            flat_models,
            self.weight_table_params,
            self.input_features
        )

        logging.info("Feature reduction applied. Creating feature file...")
        X = None
        y = []

        for _ in range(len(flat_models)):
            (model_arch, models) = flat_models.popitem()
            model_index = 0

            logging.info("Parsing %s models...", model_arch)
            for _ in tqdm(range(len(models))):
                model = models.pop(0)
                y.append(model_ground_truth_dict[model_arch][model_index])
                model_index += 1

                model_feats = use_feature_reduction_algorithm(
                    layer_transform[model_arch], model
                )
                if X is None:
                    X = model_feats
                    continue

                X = np.vstack((X, model_feats * self.model_skew["__all__"]))

        logging.info("Training RandomForestRegressor model...")
        model = RandomForestRegressor(
            **self.random_forest_kwargs,
            random_state=0
        )
        model.fit(X, y)

        logging.info("Saving RandomForestRegressor model...")
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(model, fp)

        self.write_metaparameters()
        logging.info("Configuration done!")

    def grab_inputs(self, examples_dirpath):
        inputs_np = None
        g_truths = []
        
        for examples_dir_entry in os.scandir(examples_dirpath):
            if (examples_dir_entry.is_file() and
                    examples_dir_entry.name.endswith(".npy")):
                base_example_name = os.path.splitext(
                    examples_dir_entry.name
                )[0]
                ground_truth_filename = os.path.join(
                    examples_dirpath,
                    '{}.json'.format(base_example_name)
                )
                if not os.path.exists(ground_truth_filename):
                    logging.warning(
                        f'ground truth file not found({ground_truth_filename})'
                        f'for example {base_example_name}'
                    )
                    continue

                new_input = np.load(examples_dir_entry.path)

                if inputs_np is None:
                    inputs_np = new_input
                else:
                    inputs_np = np.concatenate([inputs_np, new_input])

                with open(ground_truth_filename) as f:
                    data = int(json.load(f))

                g_truths.append(data)

        g_truths_np = np.asarray(g_truths)
        return inputs_np, g_truths_np

    def generate_statistics_datasets(
        self,
        model: DrebinNN,
        x_dataset: np.array,
        y_dataset: np.array
    ):
        '''
            Calculate main verification scores including
            ROC_AUC, f1_score, precision, recall for model
            over x_dataset, y_dataset.
            Display predicted vs Ground truth labels.
            Generate binomial perturbations and get 
            the number of produced adversarial examples

            Args:
                model - deep neural network model
                x_dataset - features
                y_dataset - labels
        '''
        # Calculate verification scores
        probabilities = generate_predictions_for_verification(
            model,
            x_dataset
        )

        results = verify_binary_classifier(
            probabilities[:, 1].cpu().detach().numpy(),
            y_dataset
        )

        #######################################################################

        # Predicted vs Ground truth labels
        count0, count1 = utils.utils.get_prediction_class_samples(
            probabilities,
            x_dataset.shape[0])
        results['pred_class0'], results['pred_class1'] = count0, count1

        count0, count1 = utils.utils.get_no_labels_class(
            y_dataset
        )
        results['actual_class0'], results['actual_class1'] = count0, count1

        #######################################################################
        # Generate binomial perturbations
        x_dataset_aug, flips = apply_binomial_pert_dataset(
            x_dataset,
            self.infer_aug_dataset_factor,
            self.infer_aug_bin_prob,
            'drebinn')
        #######################################################################
        # Find samples corresponding to perturbed samples
        # in original and augmented datasets

        list_aug_index = get_flipped_samples_indices(
            flips,
            x_dataset_aug.shape[0]
        )
        list_orig_index = [
            index % x_dataset.shape[0]
            for index in list_aug_index
        ]
        #######################################################################
        # Run model on both original and augmented datasets
        X = torch.from_numpy(x_dataset).float().to(model.device)
        output_orgset = model.model(X)

        X_aug = torch.from_numpy(x_dataset_aug).float().to(model.device)
        output_aug = model.model(X_aug)

        #######################################################################
        # Get number of adversarial examples that switched classes
        inx_adv01, inx_adv10 = identify_adversarial_examples(
            output_orgset[list_orig_index, :],
            output_aug[list_aug_index, :]
        )
        results['orig_dataset_shape'] = x_dataset.shape
        results['aug_dataset_shape'] = x_dataset_aug.shape
        results['no_adv_examples_01'] = len(inx_adv01)
        results['no_adv_examples_10'] = len(inx_adv10)

        # Specify the file name
        file_path = os.path.join(
            self.reference_model_dirpath,
            self.infer_path_adv_examples
        )
        # Writing JSON data
        with open(file_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

    def generate_adersarial_examples(
        self,
        model: DrebinNN,
        dataset: np.array
    ) -> list:
        '''
        Generates adversarial examples for a binary classifier
        and the Drebbin dataset using fast gradient sign method
        and save them to disk.
        Args:
            model: deep neural network
        Output: list of 4 torch arrays with adversarial examples
        '''

        # Calculate Jacobians
        X = torch.from_numpy(dataset).float().to(model.device)
        jacobian = get_jac(model.model, X)

        # Use fast gradient sign method to generate
        # potential adversarial examples

        X_modified_pc0, X_modified_pc1 = fast_gradient_sign_method(
            X,
            jacobian,
            model.device,
            self.infer_grad_magnitude
        )

        # Identify adversarial examples for the binary classifier
        output = model.model(X)
        output_pc0 = model.model(X_modified_pc0)
        (
            index_adversarial_examples_class01_pc0,
            index_adversarial_examples_class10_pc0
        ) = identify_adversarial_examples(output, output_pc0)

        output_pc1 = model.model(X_modified_pc0)
        (
            index_adversarial_examples_class01_pc1,
            index_adversarial_examples_class10_pc1
        ) = identify_adversarial_examples(output, output_pc1)

        return [
                X_modified_pc0[index_adversarial_examples_class01_pc0, :],
                X_modified_pc0[index_adversarial_examples_class10_pc0, :],
                X_modified_pc1[index_adversarial_examples_class01_pc1, :],
                X_modified_pc1[index_adversarial_examples_class10_pc1, :]
        ]

    def get_poison_probability(
        self,
        model: DrebinNN,
        method: str,
        agg: str,
        test_samples_path: str,
        feature_importance: bool = True,
        random_noise_augmentation: bool = True,
        date_mode: str = 'drebinn'
    ):

        """ Calculates probability that model is poisoned using different
        input samples and a reference clean model given by
        sample_model_dirpath

        Args:
            model: a pytorch model
            method: jac, discrete_deriv, shap, model_output
            agg: available options are cosine of the averaged Jacobians,
                 average of the cosine similarities of the Jacobians and
                Jensen Shannon divergence only for the model output.
            examples_dirpath: the directory path for the round example
                data used to infer if the model is poisoned or not
            drebbin_data: Drebbin dataset usage flag
            sample_model_dirpath: path for reference clean model and clean data
            date_mode: rand, real, realpert, discrete_deriv, drebinn,
                       drebinn_adversarial

        Output:
            outcome: probability for model to be poisoned
        """

        # Prepare dataset ingestions

        reference_model_path = os.path.join(
            self.reference_model_dirpath,
            'model.pt'
        )
        reference_model_samples_dirpath = os.path.join(
            self.reference_model_dirpath,
            'clean-example-data'
        )

        if date_mode == 'drebinn':

            drebbin_np, _ = get_Drebbin_dataset(
                self.reference_model_dirpath,
                self.infer_path_drebbin_x_train,
                self.infer_path_drebbin_x_test,
                self.infer_path_drebbin_y_train,
                self.infer_path_drebbin_y_test,
            )

        elif date_mode == 'drebinn_adversarial':
            
            path_adv_examples = os.path.join(
                self.reference_model_dirpath,
                self.infer_path_adv_examples
            )
            print("path_adv_examples:", path_adv_examples)
            adv_exm_class10_pc0 = np.load(os.path.join(
                path_adv_examples,
                self.infer_adv_examples_file_names[1]
                )
            )
            adv_exm_class01_pc1 = np.load(os.path.join(
                path_adv_examples,
                self.infer_adv_examples_file_names[3]
                )
            )

        elif date_mode == 'poison':
            print("path poison:", os.path.join(
                self.reference_model_dirpath,
                self.infer_path_poisoned_examples,
                self.infer_filename_poisoned_examples
                ))
            poison_examples = np.load(os.path.join(
                self.reference_model_dirpath,
                self.infer_path_poisoned_examples,
                self.infer_filename_poisoned_examples
                )
            )
        else:
            pass

        if feature_importance:
            print("Feature importance path:", os.path.join(
                self.reference_model_dirpath,
                self.infer_feature_importance_path
                ))
            feature_importance_index = np.load(os.path.join(
                self.reference_model_dirpath,
                self.infer_feature_importance_path
                )
            )
        ##################################################################

        # Load reference model samples
        inputs_np, _ = self.grab_inputs(
            reference_model_samples_dirpath
        )
        
        test_inputs_np, _ = self.grab_inputs(test_samples_path)
        # Get access to the testing server samples
        inputs_np = np.concatenate([inputs_np, test_inputs_np])
        ##################################################################

        # Dataset augmentation choices

        if date_mode == 'drebinn':

            inputs_np = np.concatenate([
               inputs_np,
               drebbin_np
               ]
            )

        elif date_mode == 'drebinn_adversarial':

            inputs_np = np.concatenate([
                inputs_np,
                adv_exm_class10_pc0,
                adv_exm_class01_pc1
                ]
            )

        elif date_mode == 'poison':
            inputs_np = np.concatenate([inputs_np, poison_examples])
        else:
            pass

        ##################################################################
        if random_noise_augmentation:
            # Apply random perturbations and augmentation 
            inputs_np, _ = apply_binomial_pert_dataset(
                inputs_np,
                self.infer_aug_dataset_factor,
                self.infer_aug_bin_prob,
                date_mode
            )
        # Load input to appropriate model.device
        X = torch.from_numpy(inputs_np).float().to(model.device)

        ##################################################################
        ##################################################################
        # Load reference clean model
        reference_model, _, _ = load_model(reference_model_path)
        # Set both reference clean model and test model to eval mode
        reference_model.model.eval()
        model.model.eval()

        ##################################################################
        # Apply poison detection methods

        if method == 'shap':

            # This method is slow and requires a lot of memory.
            # Potential memory clogs may apear if large dataset X is provided.

            if date_mode not in ['drebinn_adversarial', 'None']:
                msg = (
                    "Shap is compatible with drebinn_adversarial and None data "
                    "augmentation option!"
                )
                raise Exception(msg)
            test_features = get_shapley_values(model.model, [X], [X])   
            ref_features = get_shapley_values(reference_model.model, [X], [X])

            if feature_importance:

                test_features_least = extract_subset_features(
                    feature_importance_index,
                    test_features,
                    False,
                    self.no_features_least
                )

                ref_features_least = extract_subset_features(
                    feature_importance_index,
                    ref_features,
                    False,
                    self.no_features_least
                )

                test_features_most = extract_subset_features(
                    feature_importance_index,
                    test_features,
                    True,
                    self.no_features_most
                )

                ref_features_most = extract_subset_features(
                    feature_importance_index,
                    ref_features,
                    True,
                    self.no_features_most
                )

        ##################################################################

        elif method == 'jac':
            # Jacobian method
            test_features = get_jac(model.model, X)
            ref_features = get_jac(reference_model.model, X)

            if feature_importance:

                test_features_least = extract_subset_features(
                    feature_importance_index,
                    test_features,
                    False,
                    self.no_features_least
                )

                ref_features_least = extract_subset_features(
                    feature_importance_index,
                    ref_features,
                    False,
                    self.no_features_least
                )

                test_features_most = extract_subset_features(
                    feature_importance_index,
                    test_features,
                    True,
                    self.no_features_most
                )

                ref_features_most = extract_subset_features(
                    feature_importance_index,
                    ref_features,
                    True,
                    self.no_features_most
                )

        ##################################################################
        elif method == 'discrete_deriv':

            perturbed_inputs_np = get_discrete_derivative_inputs(inputs_np)
            perturbed_inputs_np = torch.from_numpy(perturbed_inputs_np).float().to(model.device)
            # Discrete derivatives (gradients)
            test_features = get_discrete_derivatives(model, X, perturbed_inputs_np)
            ref_features = get_discrete_derivatives(reference_model, X, perturbed_inputs_np)

            if feature_importance:

                test_features_least = extract_subset_features(
                    feature_importance_index,
                    test_features,
                    False,
                    self.no_features_least
                )

                ref_features_least = extract_subset_features(
                    feature_importance_index,
                    ref_features,
                    False,
                    self.no_features_least
                )

                test_features_most = extract_subset_features(
                    feature_importance_index,
                    test_features,
                    True,
                    self.no_features_most
                )

                ref_features_most = extract_subset_features(
                    feature_importance_index,
                    ref_features,
                    True,
                    self.no_features_most
                )
        ##################################################################
        elif method == 'model_out':
            test_features = get_scaled_model_output(model, X)
            ref_features = get_scaled_model_output(reference_model, X)
        ##################################################################
        ##################################################################

        # Apply aggregation strategies

        if agg == 'avgcos':

            if feature_importance:
                cossim_least = utils.utils.avgcosim(
                    test_features_least,
                    ref_features_least
                )
                cossim_most = utils.utils.avgcosim(
                    test_features_most,
                    ref_features_most
                )
                outcome = 0.8 * cossim_least + 0.2 * cossim_most
            else:
                outcome = utils.utils.cossim(test_features, ref_features)
        ##################################################################
        elif agg == 'cosavg':

            test_features = test_features.mean(axis=0)
            ref_features = ref_features.mean(axis=0)

            if feature_importance:

                test_features_least = test_features_least.mean(axis=0)
                ref_features_least = ref_features_least.mean(axis=0)

                test_features_most = test_features_most.mean(axis=0)
                ref_features_most = ref_features_most.mean(axis=0)

                cossim_least = utils.utils.cossim(
                    test_features_least,
                    ref_features_least
                )
                cossim_most = utils.utils.cossim(
                    test_features_most,
                    ref_features_most
                )

                outcome = 0.8 * cossim_least + 0.2 * cossim_most
            else:
                outcome = utils.utils.cossim(test_features, ref_features)
        ##################################################################
        elif agg == 'jensen-shannon':

            if method != 'model_out':
                msg = (
                    "Jensen-Shannon divergence is working only "
                    "with model output method !!!"
                )
                raise Exception(msg)

            outcome_class0 = scipy.spatial.distance.jensenshannon(
                test_features[:, 0],
                ref_features[:, 0],
                base=2
            )**2
            outcome_class1 = scipy.spatial.distance.jensenshannon(
                test_features[:, 1],
                ref_features[:, 1],
                base=2
            )**2
            outcome = 0.5*outcome_class0 + 0.5*outcome_class1
            if np.isnan(outcome): outcome = 0  
        ##############################################################################################################################
        elif agg == 'MSEavg':

            test_features = test_features.mean(axis=0)
            ref_features = ref_features.mean(axis=0)
            outcome = mean_squared_error(test_features, ref_features)
        ##############################################################################################################################
        elif agg == 'MAEavg':

            test_features = test_features.mean(axis=0)
            ref_features = ref_features.mean(axis=0)
            outcome = mean_absolute_error(test_features, ref_features)

        elif agg == 'adversarial_examples':
            if method != 'model_out':
                msg = (
                    "adversarial_examples is working only "
                    "with model output method !!!"
                )
                raise Exception(msg)
            inx_adv01, inx_adv10 = identify_adversarial_examples(
                ref_features,
                test_features
            )
            outcome = len(inx_adv01) + len(inx_adv10)

        else:
            pass

        ##############################################################################################################################
        ##############################################################################################################################
        # Derive the probability from the outcome

        return scale_probability(outcome, agg)

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        model, _, _ = load_model(model_filepath)

        if self.infer_load_drebbin:
            inputs_np, labels_np = get_Drebbin_dataset(
                self.reference_model_dirpath,
                self.infer_path_drebbin_x_train,
                self.infer_path_drebbin_x_test,
                self.infer_path_drebbin_y_train,
                self.infer_path_drebbin_y_test,
            )

        if self.train_random_forest_feature_importance:
            if not self.infer_load_drebbin:
                msg = (
                    "Set load_drebbin to True to generate statistics!"
                )
                raise Exception(msg)
            file_path = os.path.join(
                self.reference_model_dirpath,
                self.infer_feature_importance_path
            )
            get_important_features(inputs_np, labels_np, file_path)

        if self.infer_generate_statistics:
            if not self.infer_load_drebbin:
                msg = (
                    "Set load_drebbin to True to generate statistics!"
                )
                raise Exception(msg)
            self.generate_statistics_datasets(model, inputs_np, labels_np)

        if self.infer_calc_drebbin_adv:
            if not self.infer_load_drebbin:
                msg = (
                    "Set infer_load_drebbin to true to calculate adv samples for Drebbin!"
                )
                raise Exception(msg)

            list_adversarial_ex = self.generate_adersarial_examples(
                model,
                inputs_np
            )

            if self.infer_save_adv_examples:
                save_adversarial_examples_binarry_classifier(
                    os.path.join(
                        self.reference_model_dirpath,
                        self.infer_path_adv_examples,
                    ),
                    list_adversarial_ex,
                    self.infer_adv_examples_file_names,)

        probability = self.get_poison_probability(
            model,
            self.infer_feature_extraction_method,
            self.infer_proximity_aggregation_method,
            examples_dirpath,
            self.feature_importance,
            self.random_noise_augmentation,
            date_mode=self.infer_extra_data_augmentation
        )

        if self.infer_platform == 'local':
            my_dict = {get_model_name(model_filepath): probability}
            save_dictionary_to_file(my_dict, result_filepath)
        else:
            with open(result_filepath, "w") as fp:
                fp.write(probability)

        logging.info("Trojan probability: %s", probability)
