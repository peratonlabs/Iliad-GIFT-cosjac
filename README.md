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

# Setup the Conda environment

1. conda create -n r17_update python=3.7.13
2. conda activate r17_update
3. conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
4. pip install --upgrade pip
5. pip install tqdm jsonschema jsonargparse scikit-learn shap matplotlib

# Running the code outsite the Singularity container

Two pipelines configure and infer are implemented. The configure prepares the dependencies whereas infer runs the poison model detector. The default configuration outside of Singularity container can be run as described below.

(1) python entrypoint.py configure --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json  --learned_parameters_dirpath ./learned_parameters/ 


(2) python entrypoint.py infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters

These steps are described in details below. 

## (1) Run configure outside of Singularity container

The configure mode must be run first with a minimum setup. It will copy the reference model to the container.

python entrypoint.py configure --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json  --learned_parameters_dirpath ./learned_parameters/ 

In case Drebbin and Poison datasets exist, the configure mode will copy them to the container, train a surrogate random forest model and get the feature importance, generate adversarial examples and compute statistics for the reference model. To process these datasets, metaparameters infer_drebbin_dataset_exist and infer_poison_dataset_exist must be set to true. Set also to true metaparameter train_random_forest_feature_importance to generate feature importance vector, metaparameter infer_calc_drebbin_adv to generate drebbin adversarial examples and metaparameter infer_generate_statistics to calculate statistics for the reference model. 

python entrypoint.py configure --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json  --learned_parameters_dirpath ./learned_parameters/ --drebbin_dataset_dirpath ~/cyber-apk-nov2023-vectorized-drebin --poison_dataset_path ~/poison_data/

## (2) Run inference outside of Singularity container

```
python entrypoint.py infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters
```

# Build a new container - This option is needed in case you want to submit your code to TrojAI test server to evaluate your results. Set the metaparameter infer_platform to test_server.

```
sudo singularity build --force ./cyber-apk-nov2023_sts_cosjac_public.simg
```

# Container usage: Inferencing Mode - This option is needed in case you want to submit your code to TrojAI test server to evaluate your results.

```
singularity run --nv ./cyber-apk-nov2023_sts_cosjac.simg infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters
```

# Remote terminal to access google drive via API - Setup rclone to interface with google drive from remote terminal 

1. Set Up an SSH Tunnel With PuTTY
   a. Hostname – username@ml-9 
   b. Left menu – select ssh and then tunnels
   c. In Source Port tab add 53682 and in destination add localhost: 53682
   d. Open and insert password. 
2. Install rclone (I installed it on ml-9)
   a. sudo apt update
   b. sudo apt install rclone
3. In the SSH Tunnel terminal just run - rclone config
   a. Create a New Remote - When prompted, choose to create a new remote (typically by entering n).
   b. Name the Remote: Enter a name for your remote (e.g., gdrive).
   c. Storage Type: You'll be presented with a list of storage types. Enter the number or alias corresponding to Google Drive - drive.
   d. Client ID and Secret: press Enter to use the defaults.
   e. Scope: For full access to all files, choose the option with full access – 1.
   f. Root Folder ID: press Enter to use the defaults.
   g. Service Account: leave this blank (press Enter)
   h. Advanced Config: Say 'No' to advanced config.
   i. Auto-Config:  Select Yes.
   j. Copy the URL provided by rclone into a web browser.
   k. Log into your Google account - user@gmail.com
   l. Go back to the SSH window – Finish configuration by selecting No to google team or so.
   m. Test API connection with google drive
   	i.  rclone ls gdrive:
        ii. Upload container - rclone copy container_name gdrive:/

   Complete description available at - https://www.youtube.com/watch?v=n7yB1x2vhKw

# Run probability scores for all test models locally - This is useful when multiple test models are evaluated using our detector.

python run_all_models.py --test_models_path ~/cyber-apk-nov2023-train-rev2/models/ --metadata_path ~/r17_dataset/rev2/cyber-apk-nov2023-train-rev2/METADATA.csv --dictionary_path ~/r17/scratch/ --pandas_path ~/r17/scratch/output.csv

# Code capabilities

# Methods

To extract features from the tested and reference models we've calculat jacobians, shapley values, discrete derivatives and outputs. The metaparameter infer_feature_extraction_method can take any of the following options - "jac", "shap", "discrete_deriv", or "model_out". The entire set of metaparameters controlling the methods configurations is available in metaparameters.json file. 

# Proximity and aggregation methods

The detector compares the exctracted features by using one of the following proximity and aggregation methods: avgcos, cosavg, jensen-shannon, MSEavg, MAEavg, adversarial_examples. The prefered method can be set in the metaparameter infer_proximity_aggregation_method.  

# Dataset 

The detector will work on a three samples dataset provided in models/id-00000001/clean-example-data/. By default the option infer_random_noise_augmentation is set to true in the metaparameters.json and together with the metaparameter infer_aug_dataset_factor expand the three samples dataset by applying random binary perturbations. 

# Extra data augmentation methods require datasets that are not available in the repository but could be released by TrojAI organizers. 

The configure mode is required to generate feature importance dependencies and adversarial examples for Drebbin dataset and a reference model given at learned_parameters/models/id-00000001.   

python entrypoint.py configure --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json  --learned_parameters_dirpath ./learned_parameters/ --drebbin_dataset_dirpath ~/cyber-apk-nov2023-vectorized-drebin

## Running detector with Drebbin dataset

TrojAI may provide this dataset in the form of four Numpy binary files. The relative path must be provided in the metaparameters variables infer_path_drebbin_x_train, infer_path_drebbin_x_test, infer_path_drebbin_y_train, and infer_path_drebbin_y_test. The full path is obtained by merging the --reference_model_location inline argument with the metaparameters variables. The x datasets have the size of (no_samples, 991) whereas the y datasets have the size of (no_samples,). 

Full path example:

--reference_model_location /learned_parameters/models/id-00000001
infer_path_drebbin_x_train = "cyber-apk-nov2023-vectorized-drebin/x_train_sel.npy"
full_path = learned_parameters/models/id-00000001/cyber-apk-nov2023-vectorized-drebin/x_train_sel.npy

For this option, limit metaparameter infer_aug_dataset_factor to 1. The max_workers should be limited to 8 if run_all_models.py pipeline is used to evaluate multiple models.  

Running the detector with this option is enabled by setting metaparameter infer_extra_data_augmentation to "drebinn". 

## Drebin adversarial dataset 

### Calculate adversarial examples for the reference model

This option will compute the adversarial examples for the Drebbin dataset and reference model learned_parameters/models/id-00000001/model.pt. Fast gradient sign method is implemented. The adversarial examples calculation is enabled by setting metaparameters infer_load_drebbin and infer_calc_drebbin_adv to true. The function calculating the adversarial examples is infer_calc_drebbin_adv and it is called in the detector.py. The output consists of four np.ndarrays. Set the path for saving them in the metaparameters infer_path_adv_examples and infer_adv_ex_file_class01_pc0, infer_adv_ex_file_class10_pc0, infer_adv_ex_file_class01_pc1, infer_adv_ex_file_class10_pc1. It includes adversarial examples of size (no_samples, 991) that switch labels (from class 0 to 1 and class 1 to 0 denoted class01 or class10 in the files names) with respect to the class 0 or class 1 probabilities (denoted pc0 or pc1 in the files names).  

Full path example:

--reference_model_location /learned_parameters/models/id-00000001
infer_path_adv_examples =  "save_adversarial_examples"
infer_adv_examples_file_names = ["X_modified_class01_pc0.npy", "X_modified_class10_pc0.npy", "X_modified_class01_pc1.npy", "X_modified_class10_pc1.npy"]
Single file full_path = learned_parameters/models/id-00000001/save_adversarial_examples/X_modified_class01_pc0.npy

### Running detector with Drebin adversarial dataset 

Running the detector with this option is enabled by setting metaparameter infer_extra_data_augmentation to "drebinn_adversarial". Hyperparameter for the adversarial method is infer_grad_magnitude. 

## Running detector with Poison dataset

TrojAI may provide this dataset. The path to poisoned examples is defined by inline argument --reference_model_location and metaparameters infer_path_poisoned_examples and infer_filename_poisoned_examples. 

Full path example:
--reference_model_location /learned_parameters/models/id-00000001
infer_path_poisoned_examples = "poisoned_examples"
infer_filename_poisoned_examples = "poisoned_features.npy"
full_path = models/id-00000001/poisoned_examples

Running the detector with this option is enabled by setting metaparameter infer_extra_data_augmentation to "poison".

# Feature Importance
Feature importance elevates the ROC-AUC scores signficantly.  

To enable this option, one needs a dataset such as Drebbin. We trained a random forest model using the Drebbin dataset and extracted the features importance. It can be done by setting the metaparameter train_random_forest_feature_importance to true. The features importance will be saved to disk at location obtained by merging --reference_model_location and infer_feature_importance_path.

Full path example:

--reference_model_location /learned_parameters/models/id-00000001
infer_feature_importance_path = "feature_importance/index_array.npy"
full_path = learned_parameters/models/id-00000001/feature_importance/index_array.npy

Running the detector with this option requires setting infer_feature_importance metaparameter to true. 