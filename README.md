This repo is adapted from [NIST's Round 17 example code](https://github.com/usnistgov/trojai-example/tree/cyber-apk-nov2023). It contains code developed by the Perspecta Labs/Project Iliad "GIFT" team for the IARPA/TrojAI program. Code was developed by Razvan Stefanescu, Todd Huster, Peter Lin and Emmanuel Ekwedike.

Contact: razvan.stefanescu@peratonlabs.com

# Short description 
The poison model detector compares features from the potential poisoned model with features from a clean reference model.
We implemented four different features extraction methods based on jacobians, discrete derivatives, Shapley values and model outputs. To comperate and aggregate the results we used cosine similarities of averages, averages of cos similarities, jensen-shannon, MSE of the averages, MAE of the averages avg, and adversarial_examples. We also provided three extra data augmentation options based on Drebbin dataset, Drebbin adversarial and a Poisoned dataset. These options require the acquisition of the Drebbin and/or Poisoned datasets which are not provided in this repo but could potentially be released by TrojAI organizers A feature importance option is also provided in case a dataset is available for training a random forest model.

Here are the existing tested combinations.

| Method    | Cosavg | Avgcos | Jensen-Shannon | MSEavg | MAEavg | Adversarial Examples |
|-----------|--------|--------|----------------|--------|--------|----------------------|
| Jac       | Yes    | Yes    | No             | No     | No     | No                   |
| DiscreteD | Yes    | Yes    | No             | No     | No     | No                   |
| Model-Out | Yes    | Yes    | Yes            | Yes    | Yes    | Yes                  |
| Shap      | Yes    | Yes    | No             | No     | No     | No                   |

# Setup

## Create the conda environment

```
conda create -n r17_update python=3.7.13
conda activate r17_update
conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install --upgrade pip
pip install tqdm jsonschema jsonargparse scikit-learn shap matplotlib
```

## Set up data

To enable all option for TrojAI round 17, place the Drebbin dataset in the ?? directory.  Also, do x,y,z?

# Running the code outside the Singularity container

Two pipelines configure and infer are implemented. The configure prepares the dependencies whereas infer runs the poison model detector. The default configuration outside of Singularity container can be run as described below.

(1) python entrypoint.py configure --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json  --learned_parameters_dirpath ./learned_parameters/ 


(2) python entrypoint.py infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters

These steps are described in details below. 

## (1) Run configure outside of Singularity container

The configure mode must be run first with a minimum setup. It will copy the reference model to the container.

python entrypoint.py configure --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json  --learned_parameters_dirpath ./learned_parameters/ 

In case Drebbin and Poison datasets exist, the configure mode will copy them to the container, train a surrogate random forest model and get the feature importance, generate adversarial examples and compute statistics for the reference model. Set up the --drebbin_dataset_dirpath and --poison_dataset_path accordingly with appropriate paths. To process these datasets, metaparameters infer_drebbin_dataset_exist and infer_poison_dataset_exist must be set to true. Set also to true metaparameter train_random_forest_feature_importance to generate feature importance vector, metaparameter infer_calc_drebbin_adv to generate drebbin adversarial examples and metaparameter infer_generate_statistics to calculate statistics for the reference model. 

python entrypoint.py configure --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json  --learned_parameters_dirpath ./learned_parameters/ --drebbin_dataset_dirpath ~/cyber-apk-nov2023-vectorized-drebin --poison_dataset_path ~/poison_data/

## (2) Run inference outside of Singularity container

```
python entrypoint.py infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters
```

# Build a new container

This option is needed in case you want to submit your code to TrojAI test server to evaluate your results. Set the metaparameter infer_platform to test_server.

```
sudo singularity build --force ./cyber-apk-nov2023_sts_cosjac_public.simg
```


# Container usage: Inferencing Mode


This option is needed in case you want to submit your code to TrojAI test server to evaluate your results. 

```
singularity run --nv ./cyber-apk-nov2023_sts_cosjac_public.simg infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters
```

# Run probability scores for all test models locally

This is useful when multiple test models are evaluated using our detector. Access to a collection of models is needed. Set up the path for the test models in --test_models_path. Don't forget to set the metaparameter infer_platform to local since it can only run on the local system. With the current setup, the system must have two GPUs. The distributed implementation requires locals tunning for no_available_GPUs and max_workers (number of threads). 

python run_all_models.py --test_models_path ~/cyber-apk-nov2023-train-rev2/models/ --metadata_path ~/cyber-apk-nov2023-train-rev2/METADATA.csv --dictionary_path ~/r17/scratch/ --pandas_path ~/r17/scratch/output.csv

## Run probability scores for all test models locally for multiple metaparameters configurations

To run multiple metaparameters configurations, the user must specify the metaparameters folder with metaparameters.json files, test_models_path, metadata file containing models poison information and output_folder. A collection of metaparameters files is available in the repository at metaparameters_configs.

python run_multi_metaparameters_experiments.py --metaparameters_folder metaparameters_configs/extra_data_augmentation_None/ --test_models_path ~/cyber-apk-nov2023-train-rev2/models/ --metadata_path  ~/cyber-apk-nov2023-train-rev2/METADATA.csv --output_folder scratch/

# Code capabilities

# Methods

To extract features from the tested and reference models we've calculat jacobians, shapley values, discrete derivatives and outputs. The metaparameter infer_feature_extraction_method can take any of the following options - "jac", "shap", "discrete_deriv", or "model_out". The entire set of metaparameters controlling the methods configurations is available in metaparameters.json file. 

# Proximity and aggregation methods

The detector compares the exctracted features by using one of the following proximity and aggregation methods: avgcos, cosavg, jensen-shannon, MSEavg, MAEavg, adversarial_examples. The prefered method can be set in the metaparameter infer_proximity_aggregation_method.  

Check the table above for the tested options. 

# Dataset 

The detector will work on a three samples dataset provided in models/id-00000001/clean-example-data/. By default the option infer_random_noise_augmentation is set to true in the metaparameters.json and together with the metaparameter infer_aug_dataset_factor expand the three samples dataset by applying random binary perturbations. 

# Extra data augmentation methods. 

To run the code using these options, the Drebbin dataset and/or Poison dataset must be available. Check the (1) Run configure outside of Singularity container section above on how to prepare the dependencies. The following options are available ["drebinn", "drebinn_adversarial", "poison"] in addition to the default "None". Set metaparameter infer_extra_data_augmentation accordingly. 

## Running detector with Drebbin dataset

Drebbin dataset contains four Numpy binary files. Limit metaparameter infer_aug_dataset_factor to 1 since the dataset has a large number of samples. The max_workers should be limited to 8 if run_all_models.py pipeline is used to evaluate multiple models.  

Running the detector with this option is enabled by setting metaparameter infer_extra_data_augmentation to "drebinn".

## Running detector with Drebin adversarial dataset 

The adversarial examples for the Drebbin dataset and reference model learned_parameters/models/id-00000001/model.pt are required. Section (1) Run configure outside of Singularity container explains how to generate them using fast gradient sign method.

Running the detector with this option is enabled by setting metaparameter infer_extra_data_augmentation to "drebinn_adversarial". The max_workers should be limited to 16 if run_all_models.py pipeline is used to evaluate multiple models. 

## Running detector with Poison dataset

If poison dataset is available, Section (1) Run configure outside of Singularity container explains how to copy the data to the appropriate location. 

Running the detector with this option is enabled by setting metaparameter infer_extra_data_augmentation to "poison".

# Feature Importance
Feature importance improves the ROC-AUC scores signficantly. It requires the Drebbin dataset and a random forest was trained to calculate the feature importance in  Section (1) Run configure outside of Singularity container. 
Running the detector with this option requires setting infer_feature_importance metaparameter to true. 
