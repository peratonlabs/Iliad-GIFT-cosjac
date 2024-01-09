import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import shap

from tqdm import tqdm
import torch
import torch.nn.functional as F
import utils.utils
from utils.utils import get_shapley_values, verify_binary_classifier
from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath, build_random_forest_classifier
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
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

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

    def write_metaparameters(self):
        metaparameters = {
            "infer_cyber_model_skew": self.model_skew["__all__"],
            "train_input_features": self.input_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        models_padding_dict = create_models_padding(model_repr_dict)
        with open(self.models_padding_dict_filepath, "wb") as fp:
            pickle.dump(models_padding_dict, fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

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

        layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.input_features)

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
        model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
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
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                base_example_name = os.path.splitext(examples_dir_entry.name)[0]
                ground_truth_filename = os.path.join(examples_dirpath, '{}.json'.format(base_example_name))
                if not os.path.exists(ground_truth_filename):
                    logging.warning('ground truth file not found ({}) for example {}'.format(ground_truth_filename, base_example_name))
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

    def inference_on_example_data(self, model, method: str, agg: str, 
                                  examples_dirpath, drebinn_data: bool, sample_model_dirpath='/models/id-00000001', mode='real'):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        
        
        try: 
            sample_model_path = os.path.join(sample_model_dirpath, 'model.pt')
            sample_model_examples_dirpath = os.path.join(sample_model_dirpath, 'clean-example-data')
            sample_model, _, _ = load_model(sample_model_path)
            if drebinn_data:
                feature_importance_index = np.load(os.path.join(sample_model_dirpath, 'feature_importance/index_array.npy'))
            #    drebinn_x_train = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/x_train_sel.npy'))
            #    drebinn_y_train = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/y_train_sel.npy'))
            #    drebinn_x_test = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/x_test_sel.npy'))
            #    drebinn_y_test = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/y_test_sel.npy'))        
        except:
            sample_model_dirpath = '.' + sample_model_dirpath
            print(sample_model_dirpath, "not found. Trying",sample_model_dirpath)
            sample_model_path = os.path.join('.',sample_model_dirpath, 'model.pt')
            sample_model_examples_dirpath = os.path.join('.',sample_model_dirpath, 'clean-example-data')
            sample_model, _, _ = load_model(sample_model_path)
            if drebinn_data:
                feature_importance_index = np.load(os.path.join(sample_model_dirpath, 'feature_importance/index_array.npy'))
            #    drebinn_x_train = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/x_train_sel.npy'))
            #    drebinn_y_train = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/y_train_sel.npy'))
            #    drebinn_x_test = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/x_test_sel.npy'))
            #    drebinn_y_test = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/y_test_sel.npy'))
        
        print("sample_model_path:", sample_model_path)

        if mode=='rand':
            print("sample_model_examples_dirpath:", sample_model_examples_dirpath)
            inputs_np, _ = self.grab_inputs(sample_model_examples_dirpath)
            #inputs_np = np.random.randn(1000,*inputs_np.shape[1:])
            inputs_np = np.random.randint(2,size=[1000, inputs_np.shape[1]])
            #print(inputs_np.shape)
        
        elif mode=='real' or mode=='realpert':
            
            print("examples_dirpath:", examples_dirpath)
            inputs_np, _ = self.grab_inputs(examples_dirpath)
            print("inputs np shape:", inputs_np.shape)
            print("sample_model_examples_dirpath:", sample_model_examples_dirpath)
            my_inputs_np, _ = self.grab_inputs(sample_model_examples_dirpath)
            print("my_inputs_np shape:", my_inputs_np.shape)


            inputs_np = np.concatenate([inputs_np, my_inputs_np])
            #inputs_np = my_inputs_np
            #g_truths_np = np.concatenate([g_truths_np,my_g_truths_np])
            
            if mode=='realpert':
                n_repeats = 100
                p=0.01
            
                inputs_np = np.repeat(inputs_np, n_repeats, axis=0)
                #print(inputs_np.shape, inputs_np.dtype)
                
                
                flips = np.random.binomial(1, p, size=inputs_np.shape)                
                inputs_np[flips==1] = 1 - inputs_np[flips==1]
                        
        elif mode == 'discrete_deriv':

            inputs_np, _ = self.grab_inputs(examples_dirpath)
            
            list_new_vectors = []
            for i in range(inputs_np.shape[0]):
                list_new_vectors.append(np.abs(np.eye(inputs_np.shape[1]) - inputs_np[i,:]))
            perturbed_inputs_np = np.concatenate(list_new_vectors, axis=0)

        #if drebinn_data:
            #rf_model = build_random_forest_classifier(drebinn_x_train, drebinn_y_train)
            #feature_names = [f'feature_{i}' for i in range(drebinn_x_train.shape[1])]
            #importances = rf_model.feature_importances_
            #importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            # Sort the DataFrame by importance
            #importance_df.sort_values(by='Importance', ascending=False, inplace=True)
            #print(importance_df.index)
            # Convert the DataFrame index to a NumPy array
            #index_array = importance_df.index.to_numpy()

            # Save the NumPy array to disk
            #np.save('index_array.npy', index_array)

        sample_model.model.eval()
        model.model.eval()
        
#        if drebinn_data:
#            drebinn_x_train = torch.from_numpy(drebinn_x_train).float().to(model.device)
#            output = model.model(drebinn_x_train)
#            count_label1 = 0
#            count_label2 = 0
#            for val in drebinn_y_train:
#                if val == 0:
#                    count_label1 += 1
#                else:
#                    count_label2 += 1
            # Apply softmax to convert logits to probabilities
#            probabilities = F.softmax(output, dim=1)
#            results = verify_binary_classifier(probabilities[:,1].cpu().detach().numpy(), drebinn_y_train)
#            print("Training set:", results)

#            drebinn_x_test = torch.from_numpy(drebinn_x_test).float().to(model.device)
#            output = model.model(drebinn_x_test)
#            probabilities = F.softmax(output, dim=1)
#            results = verify_binary_classifier(probabilities[:,1].cpu().detach().numpy(), drebinn_y_test)
#            print("Testing set:", results)

        if mode == 'discrete_deriv':
            X1 = torch.from_numpy(inputs_np).float().to(model.device)
            X2 = torch.from_numpy(perturbed_inputs_np).float().to(model.device) 
        else:
            X = torch.from_numpy(inputs_np).float().to(model.device)


        if method == 'shap':
            test_feature = get_shapley_values(model.model, [X], [X])        
            print("Shape of the shap values arrays:", test_feature[0].shape, test_feature[1].shape)
            print("Type of shap_values_model:", type(test_feature), type(test_feature[0]))
            test_features = np.stack(test_feature, axis=2)
        
            ref_features = get_shapley_values(sample_model.model, [X], [X])
            print("Shape of the shap values arrays:", ref_features[0].shape, ref_features[1].shape)
            ref_features = np.stack(ref_features,  axis=2)
        
        elif method == 'jac':
            test_features =  utils.utils.get_jac(model.model, X )
            ref_features =  utils.utils.get_jac(sample_model.model, X )
            if drebinn_data: 
                test_features_least = test_features[:, feature_importance_index[-80:], :]
                ref_features_least = ref_features[:, feature_importance_index[-80:], :]

                test_features_most = test_features[:, feature_importance_index[0:20], :]
                ref_features_most = ref_features[:, feature_importance_index[0:20], :]

        elif method == 'discrete_deriv':
            
            output_ref = model.model(X1)
            output_perturbed_ref = model.model(X2)
            dim1 = output_ref.shape[0]
            dim2 = X1.shape[1]
            
            discrete_derivatives = []
            for i in range(dim1):
                discrete_derivatives.append(output_ref[i,:] - output_perturbed_ref[i*dim2:(i+1)*dim2])
            test_features = torch.stack(discrete_derivatives, dim=0)

            output = sample_model.model(X1)
            output_perturbed = sample_model.model(X2)
            discrete_derivatives = []
            for i in range(dim1):
                discrete_derivatives.append(output[i,:] - output_perturbed[i*dim2:(i+1)*dim2])
            ref_features = torch.stack(discrete_derivatives, dim=0)

            test_features = test_features.cpu().detach().numpy()
            ref_features = ref_features.cpu().detach().numpy()



        if agg=='avgcos':
            #cossims = [utils.utils.cossim(test_features[i], ref_features[i])   for i in range(test_features.shape[0]) ]
            cossim = utils.utils.cossim(test_features, ref_features)
        elif agg=='cosavg':
            test_features = test_features.mean(axis=0)
            ref_features = ref_features.mean(axis=0)
            if drebinn_data:
                
                test_features_least = test_features_least.mean(axis=0)
                ref_features_least = ref_features_least.mean(axis=0)

                test_features_most = test_features_most.mean(axis=0)
                ref_features_most = ref_features_most.mean(axis=0)

                cossim_least = utils.utils.cossim(test_features_least, ref_features_least)
                cossim_most = utils.utils.cossim(test_features_most, ref_features_most)

                cossim = 0.8* cossim_least + 0.2 * cossim_most

            else:
                cossim = utils.utils.cossim(test_features, ref_features)

        #cossims = [utils.utils.cossim(test_feature[i], ref_features[i])   for i in range(test_feature.shape[0]) ]
        #cossim = np.mean(cossims)

        return cossim


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

        print("Possible reference model - filepath:", model_filepath)
        model, model_repr, model_class = load_model(model_filepath)
        model_repr = pad_model(model_repr, model_class, models_padding_dict)
        flat_model = flatten_model(model_repr, model_layer_map[model_class])

        print("examples_dirpath:", examples_dirpath)
        # Inferences on examples to demonstrate how it is done for a round
        # This is not needed for the random forest classifier
        #cossim1 = self.inference_on_example_data(model, examples_dirpath, mode='rand')
        
        #cossim2 = self.inference_on_example_data(model, examples_dirpath, mode='real')
        
        #cossim3 = self.inference_on_example_data(model, 'discrete_deriv','cosavg', examples_dirpath, mode='discrete_deriv') 
        cossim3 = self.inference_on_example_data(model, 'jac','cosavg', examples_dirpath, True, mode='realpert') 
        #cossim3 = self.inference_on_example_data(model, 'shap','cosavg', examples_dirpath, mode='realpert') 
        #cossim3 = self.inference_on_example_data(model, 'shap','avgcos', examples_dirpath, mode='realpert') 
        
        #probability = 0.5 - 0.05*cossim1  - 0.05*cossim2
        #probability = 0.5 - 0.1*cossim1  - 0.0*cossim2
        #probability = 0.5 - 0.0*cossim1  - 0.1*cossim2
        probability = 0.5 - 0.1*cossim3
        probability = str(probability)
        

        #X = (
        #    use_feature_reduction_algorithm(layer_transform[model_class], flat_model)
        #    * self.model_skew["__all__"]
        #)

        #try:
        #    with open(self.model_filepath, "rb") as fp:
        #        regressor: RandomForestRegressor = pickle.load(fp)

        #    probability = str(regressor.predict(X)[0])
        #except Exception as e:
        #    logging.info('Failed to run regressor, there may have an issue during fitting, using random for trojan probability: {}'.format(e))
        #    probability = str(np.random.rand())
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
