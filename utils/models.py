from collections import OrderedDict
import json
import numpy as np
import os
from os.path import join
import re
from sklearn.ensemble import RandomForestClassifier
import torch
from tqdm import tqdm
from utils.drebinnn import DrebinNN


def create_layer_map(model_repr_dict):
    model_layer_map = {}
    for (model_class, models) in model_repr_dict.items():
        layers = models[0]
        layer_names = list(layers.keys())
        base_layer_names = list(
            dict.fromkeys(
                [
                    re.sub(
                        "\\.(weight|bias|running_(mean|var)|num_batches_tracked)",
                        "",
                        item,
                    )
                    for item in layer_names
                ]
            )
        )
        layer_map = OrderedDict(
            {
                base_layer_name: [
                    layer_name
                    for layer_name in layer_names
                    if re.match(f"{base_layer_name}.+", layer_name) is not None
                ]
                for base_layer_name in base_layer_names
            }
        )
        model_layer_map[model_class] = layer_map

    return model_layer_map


def load_model(model_filepath: str) -> (dict, str):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to model.pt file

    Returns:
        model, dict, str - Torch model + dictionary representation of the model + model class name
    """

    conf_filepath = os.path.join(os.path.dirname(model_filepath), 'reduced-config.json')
    with open(conf_filepath, 'r') as f:
        full_conf = json.load(f)

    model = DrebinNN(991, full_conf)
    model.load('.', model_filepath)
    # model = torch.load(model_filepath)
    model_class = model.model.__class__.__name__
    model_repr = OrderedDict(
        {layer: tensor.cpu().numpy() for (layer, tensor) in model.model.state_dict().items()}
    )

    return model, model_repr, model_class


def load_ground_truth(model_dirpath: str):
    """Returns the ground truth for a given model.

    Args:
        model_dirpath: str -

    Returns:

    """

    with open(join(model_dirpath, "ground_truth.csv"), "r") as fp:
        model_ground_truth = fp.readlines()[0]

    return int(model_ground_truth)


def load_models_dirpath(models_dirpath):
    model_repr_dict = {}
    model_ground_truth_dict = {}

    for model_path in tqdm(models_dirpath):
        model, model_repr, model_class = load_model(
            join(model_path, "model.pt")
        )
        model_ground_truth = load_ground_truth(model_path)

        # Build the list of models
        if model_class not in model_repr_dict.keys():
            model_repr_dict[model_class] = []
            model_ground_truth_dict[model_class] = []

        model_repr_dict[model_class].append(model_repr)
        model_ground_truth_dict[model_class].append(model_ground_truth)

    return model_repr_dict, model_ground_truth_dict


def build_random_forest_classifier(x_train: np.array, y_train: np.array) -> RandomForestClassifier:
    '''Build a simple random forest model using scikit-library
       Input:
            x_train: np.array with the features
            y_train: labels
        Output:
            rf_model object
    '''
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_model.fit(x_train, y_train)
    return rf_model

#inputs_np = np.load(os.path.join(sample_model_dirpath, 'cyber-apk-nov2023-vectorized-drebin/x_train_sel.npy'))