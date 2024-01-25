import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import scipy
from tqdm import tqdm
import torch
import torch.nn.functional as F
import utils.utils
from utils.utils import (
    get_shapley_values, 
    verify_binary_classifier,
    apply_binomial_pert_dataset,
    get_discrete_derivative_inputs,
    get_jac,
    get_discrete_derivatives,
    get_scaled_model_output,
    scale_probability,
    fast_gradient_sign_method,
    identify_adversarial_examples
)
from utils.abstract import AbstractDetector
from utils.drebinnn import DrebinNN
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
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

        # TODO: Update skew parameters per round
        self.model_skew = {
            "__all__": metaparameters["infer_cyber_model_skew"],
        }

        self.input_features = metaparameters["train_input_features"]
        self.weight_table_params = {
            "random_seed": metaparameters["train_weight_table_random_state"],
            "mean": metaparameters["train_weight_table_params_mean"],
            "std": metaparameters["train_weight_table_params_std"],
            "scaler": metaparameters["train_weight_table_params_scaler"],
        }
        self.random_forest_kwargs = {
            "n_estimators": metaparameters[
                "train_random_forest_regressor_param_n_estimators"
            ],
            "criterion": metaparameters[
                "train_random_forest_regressor_param_criterion"
            ],
            "max_depth": metaparameters[
                "train_random_forest_regressor_param_max_depth"
            ],
            "min_samples_split": metaparameters[
                "train_random_forest_regressor_param_min_samples_split"
            ],
            "min_samples_leaf": metaparameters[
                "train_random_forest_regressor_param_min_samples_leaf"
            ],
            "min_weight_fraction_leaf": metaparameters[
                "train_random_forest_regressor_param_min_weight_fraction_leaf"
            ],
            "max_features": metaparameters[
                "train_random_forest_regressor_param_max_features"
            ],
            "min_impurity_decrease": metaparameters[
                "train_random_forest_regressor_param_min_impurity_decrease"
            ],
        }
        self.path_adv_examples = metaparameters['path_adv_examples']

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

    def generate_predictions_for_verification(
            self,
            model,
            dataset: np.ndarray
    ) -> torch:
        '''
            Generate softmax predictions
            Input: model - torch mapping
                   dataset - model input in numpy format
            Output:
                    Softmax predictions in torch format
        '''
        train = torch.from_numpy(dataset).float().to(model.device)
        output_train = model.model(train)
        return F.softmax(output_train, dim=1)
    
    def generate_statistics_datasets(
        self,
        model,
        examples_dirpath,
        drebinn_data: bool,
        sample_model_dirpath='/models/id-00000001'
    ):

        try: 
            sample_model_path = os.path.join(sample_model_dirpath, 'model.pt')
            sample_model_examples_dirpath = os.path.join(sample_model_dirpath, 'clean-example-data')
            sample_model, _, _ = load_model(sample_model_path)
            if drebinn_data:
                feature_importance_index = np.load(os.path.join(sample_model_dirpath, 'feature_importance/index_array.npy')) 
            #if mode == 'drebinn':
            drebinn_x_train = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/x_train_sel.npy'))
            drebinn_y_train = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/y_train_sel.npy'))
            drebinn_x_test = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/x_test_sel.npy'))
            drebinn_y_test = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/y_test_sel.npy'))
        except:
            sample_model_dirpath = '.' + sample_model_dirpath
            print(sample_model_dirpath, "not found. Trying",sample_model_dirpath)
            sample_model_path = os.path.join('.',sample_model_dirpath, 'model.pt')
            sample_model_examples_dirpath = os.path.join('.',sample_model_dirpath, 'clean-example-data')
            sample_model, _, _ = load_model(sample_model_path)
            if drebinn_data:
                feature_importance_index = np.load(os.path.join(sample_model_dirpath, 'feature_importance/index_array.npy'))
            #if mode == 'drebinn':
            drebinn_x_train = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/x_train_sel.npy'))
            drebinn_y_train = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/y_train_sel.npy'))
            drebinn_x_test = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/x_test_sel.npy'))
            drebinn_y_test = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/y_test_sel.npy'))

        #inputs_np, _ = self.grab_inputs(examples_dirpath)

        # Get scores for Drebinn training and testing datasets
            
        indices = np.where(drebinn_y_train == 1)[0]
        print("indices shape:", indices.shape)
        print("Shape of the random samples dataset:", drebinn_x_train[indices[np.random.randint(0, indices.shape, 3)],:].shape)
        np.save("three_uniform_random_drebbin_samples.npy", drebinn_x_train[np.random.randint(0, indices.shape, 3),:])

        no_labels_class0, no_labels_class1  = utils.utils.get_no_labels_class(drebinn_y_train)
        print("Drebbin training dataset - No of samples in class 0:", no_labels_class0)
        print("Drebbin training dataset - No of samples in class 1:", no_labels_class1)

        #output_training = model.model(torch.from_numpy(drebinn_x_train).float().to(model.device))

        probabilities = self.generate_predictions_for_verification(model, drebinn_x_train)
        results_train = verify_binary_classifier(probabilities[:,1].cpu().detach().numpy(), drebinn_y_train)
        print("Training set:", results_train)

        count0, count1 = utils.utils.get_prediction_class_samples(probabilities, drebinn_x_train.shape[0])
        print("Predicted classes 0: ", count0, " classes 1: ", count1)

        no_labels_class0, no_labels_class1  = utils.utils.get_no_labels_class(drebinn_y_test)
        print("Drebbin training dataset - No of samples in class 0:", no_labels_class0)
        print("Drebbin training dataset - No of samples in class 1:", no_labels_class1)

        probabilities = self.generate_predictions_for_verification(model, drebinn_x_test)
        results_test = verify_binary_classifier(probabilities[:,1].cpu().detach().numpy(), drebinn_y_test)
        print("Testing set:", results_test)

        #output_testing = model.model(torch.from_numpy(drebinn_x_test).float().to(model.device))
        count0, count1 = utils.utils.get_prediction_class_samples(probabilities, drebinn_x_test.shape[0])
        print("Predicted classes 0: ", count0, " classes 1: ", count1)


        inputs_np = np.concatenate([drebinn_x_train, drebinn_x_test])
        no_samples_original, _ = inputs_np.shape
        #indices_class1 = np.argwhere(label_np == 1)


        X = torch.from_numpy(inputs_np).float().to(model.device)
        output_original = model.model(X)

        #count0, count1 = 0, 0
        #for inx in range(no_samples_original):
        #    p0, p1 = output_original[inx]
        #    count0, count1 = (count0 + 1, count1) if p0 > p1 else (count0, count1 + 1)
        
        #print("Predicted classes 0: ", count0, " classes 1: ", count1)

        n_repeats = 10
        p=0.009
        print("p is:", p)

        inputs_np = np.repeat(inputs_np, n_repeats, axis=0)
        no_samples, _= inputs_np.shape        
        flips = np.random.binomial(1, p, size=inputs_np.shape)
        inputs_np[flips==1] = 1 - inputs_np[flips==1]

        list_original_dataset_index = []
        list_binomial_modified_dataset_index = []

        for inx in range(no_samples):                 
            no_changes = sum(1 for val in flips[inx,:] if val == 1)
            if no_changes > 0:
                list_original_dataset_index.append(inx%no_samples_original)
                list_binomial_modified_dataset_index.append(inx)
            
        X = torch.from_numpy(inputs_np).float().to(model.device)
        output = model.model(X)
        
        counter_switch_classes_01, counter_switch_classes_10 = 0, 0
        for inx, index in enumerate(list_binomial_modified_dataset_index):
            p0, p1 = output[index]
            p0_orig, p1_orig = output_original[list_original_dataset_index[inx],:]
            if p0 > p1 and p0_orig < p1_orig:
                counter_switch_classes_01 +=1
            elif p0 < p1 and p0_orig > p1_orig:
                counter_switch_classes_10 +=1

        print("counter_switch_classes_01:", counter_switch_classes_01)
        print("counter_switch_classes_10:", counter_switch_classes_10)

    def generate_adersarial_examples(
        self,
        model,
        examples_dirpath,
        reference_model_dirpath: str = '/models/id-00000001',
    ):
        
        # Prepare dataset ingestions

        reference_model_samples_dirpath = os.path.join(
            reference_model_dirpath,
            'clean-example-data'
        )
        ss = os.path.join(
            reference_model_dirpath,
            'cyber-apk-nov2023-vectorized-drebin/x_train_sel.npy'
            )
        print("sssssssss:", ss)
        drebinn_x_train = np.load(os.path.join(
            reference_model_dirpath,
            'cyber-apk-nov2023-vectorized-drebin/x_train_sel.npy'
            )
        )
        drebinn_x_test = np.load(os.path.join(
            reference_model_dirpath,
            'cyber-apk-nov2023-vectorized-drebin/x_test_sel.npy'
            )
        )

        # Load reference model samples
        inputs_np, _ = self.grab_inputs(
            reference_model_samples_dirpath
        )
        # Load test model samples
        test_inputs_np, _ = self.grab_inputs(examples_dirpath)
        inputs_np = np.concatenate([
            inputs_np,
            test_inputs_np,
            drebinn_x_train,
            drebinn_x_test
            ]
        )
        
        X = torch.from_numpy(inputs_np).float().to(model.device)
        jacobian = get_jac(model.model, X)

        X_modified_pc0, X_modified_pc1 = fast_gradient_sign_method(
            X,
            jacobian,
            model.device,
            0.01
        )

        (
            index_adversarial_examples_class01_pc0,
            index_adversarial_examples_class10_pc0
        ) = identify_adversarial_examples(model, X, X_modified_pc0)
        
        (
            index_adversarial_examples_class01_pc1,
            index_adversarial_examples_class10_pc1
        ) = identify_adversarial_examples(model, X, X_modified_pc1)
            
        file_class01_pc0 = self.path_adv_examples + 'X_modified_class01_pc0.npy'
        file_class10_pc0 = self.path_adv_examples + 'X_modified_class10_pc0.npy'
        file_class01_pc1 = self.path_adv_examples + 'X_modified_class01_pc1.npy'
        file_class10_pc1 = self.path_adv_examples + 'X_modified_class10_pc1.npy'
            
        np.save(file_class01_pc0, X_modified_pc0[index_adversarial_examples_class01_pc0, :].cpu().detach().numpy())
        np.save(file_class10_pc0, X_modified_pc0[index_adversarial_examples_class10_pc0, :].cpu().detach().numpy())
        np.save(file_class01_pc1, X_modified_pc1[index_adversarial_examples_class01_pc1, :].cpu().detach().numpy())
        np.save(file_class10_pc1, X_modified_pc1[index_adversarial_examples_class10_pc1, :].cpu().detach().numpy())

    def get_poison_probability(
        self,
        model: DrebinNN,
        method: str,
        agg: str,
        examples_dirpath: str,
        feature_importance: bool = True,
        reference_model_dirpath: str = '/models/id-00000001',
        date_mode: str = 'real'
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
            reference_model_dirpath,
            'model.pt'
        )
        reference_model_samples_dirpath = os.path.join(
            reference_model_dirpath,
            'clean-example-data'
        )

        if date_mode == 'drebinn':
            drebinn_x_train = np.load(os.path.join(
                reference_model_dirpath,
                'cyber-apk-nov2023-vectorized-drebin/x_train_sel.npy'
                )
            )
            drebinn_x_test = np.load(os.path.join(
                reference_model_dirpath,
                'cyber-apk-nov2023-vectorized-drebin/x_test_sel.npy'
                )
            )

        if date_mode == 'drebinn_adversarial':
            adv_exm_class10_pc0 = np.load(os.path.join(
                reference_model_dirpath,
                "adversarial_examples/X_modified_class10_pc0.npy"
                )
            )
            adv_exm_class01_pc1 = np.load(os.path.join(
                reference_model_dirpath,
                "adversarial_examples/X_modified_class01_pc1.npy"
                )
            )

        if feature_importance:
            feature_importance_index = np.load(os.path.join(
                reference_model_dirpath,
                'feature_importance/index_array.npy'
                )
            )
        ##################################################################

        # Load reference model samples
        inputs_np, _ = self.grab_inputs(
            reference_model_samples_dirpath
        )
        test_inputs_np, _ = self.grab_inputs(examples_dirpath)
        # Get access to the testing server samples
        inputs_np = np.concatenate([inputs_np, test_inputs_np])
        ##################################################################
        # Dataset augmentation choices

        if date_mode == 'drebinn':

            inputs_np = np.concatenate([
                inputs_np,
                drebinn_x_train,
                drebinn_x_test
                ]
            )

        elif date_mode == 'drebinn_reference_adversarial':

            inputs_np = np.concatenate([
                inputs_np,
                adv_exm_class10_pc0,
                adv_exm_class01_pc1
                ]
            )

        else:
            pass

        ##################################################################
        # For all data options we apply random perturbations.
        inputs_np = apply_binomial_pert_dataset(
            inputs_np,
            100,
            0.009,
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

            if date_mode != 'drebinn_reference_adversarial':
                msg = (
                    "Shap is compatible with drebinn_reference_adversarial "
                    "date_mode option!"
                )
                raise Exception(msg)
            test_features = get_shapley_values(model.model, [X], [X])   
            ref_features = get_shapley_values(reference_model.model, [X], [X])

        ##################################################################

        elif method == 'jac':
            # Jacobian method
            test_features = get_jac(model.model, X)
            ref_features = get_jac(reference_model.model, X)

            if feature_importance:

                test_features_least = test_features[
                    :,
                    feature_importance_index[-80:],
                    :
                ]
                ref_features_least = ref_features[
                    :,
                    feature_importance_index[-80:],
                    :
                ]

                test_features_most = test_features[
                    :,
                    feature_importance_index[0:20],
                    :
                ]
                ref_features_most = ref_features[
                    :,
                    feature_importance_index[0:20],
                    :
                ]

        ##################################################################
        elif method == 'discrete_deriv':
            
            perturbed_inputs_np = get_discrete_derivative_inputs(inputs_np)
            perturbed_inputs_np = torch.from_numpy(perturbed_inputs_np).float().to(model.device)
            # Discrete derivatives (gradients)
            test_features = get_discrete_derivatives(model, X, perturbed_inputs_np)
            ref_features = get_discrete_derivatives(ref_features, X, perturbed_inputs_np)

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
        ##############################################################################################################################
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

        with open(self.model_layer_map_filepath, "rb") as fp:
            model_layer_map = pickle.load(fp)

        # List all available model and limit to the number provided
        model_path_list = sorted(
            [
                join(round_training_dataset_dirpath, 'models', model)
                for model in listdir(join(round_training_dataset_dirpath, 'models'))
            ]
        )
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, _ = load_models_dirpath(model_path_list)
        logging.info("Loaded models. Flattenning...")

        with open(self.models_padding_dict_filepath, "rb") as fp:
            models_padding_dict = pickle.load(fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        # Flatten model
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.input_features)

        model, model_repr, model_class = load_model(model_filepath)
        model_repr = pad_model(model_repr, model_class, models_padding_dict)
        flat_model = flatten_model(model_repr, model_layer_map[model_class])
        #self.generate_adersarial_examples(model, examples_dirpath)
        probability = self.get_poison_probability(model, 'jac','cosavg', examples_dirpath, True, date_mode='realpert') 

        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
