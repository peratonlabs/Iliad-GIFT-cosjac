This repo is adapted from [NIST's Round 17 example code](https://github.com/usnistgov/trojai-example/tree/cyber-apk-nov2023). 


# Setup the Conda environment

1. conda create -n r17_update python=3.7.13
2. conda activate r17_update
3. conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
4. pip install --upgrade pip
5.  install tqdm jsonschema jsonargparse scikit-learn shap matplotlib

# Run inference outside of Singularity

```
python entrypoint.py infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters
```

# Build a new container 

```
sudo singularity build --force ./cyber-apk-nov2023_sts_cosjac.simg example_trojan_detector.def
```

# Container usage: Inferencing Mode - What is the difference between running outside of Singularity and this option?

```
singularity run --nv ./cyber-apk-nov2023_sts_cosjac.simg infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters
```

# Remote terminal to access google drive via API - Setup rclone to interface with google drive from remote terminal 

1. Set Up an SSH Tunnel With PuTTY
   a. Hostname – username@ml-9 (rstefanescu@ml-9)
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
   k. Log into your Google account - perspecta.gift@gmail.com
   l. Go back to the SSH window – Finish configuration by selecting No to google team or so.
   m. Test API connection with google drive
   	i.  rclone ls gdrive:
        ii. Upload container - rclone copy container_name gdrive:/

   Complete description available at - https://www.youtube.com/watch?v=n7yB1x2vhKw

# Run probability scores for all test models locally

python run_all_models.py --test_models_path /home/rstefanescu/r17_dataset/rev2/cyber-apk-nov2023-train-rev2/models/ --metadata_path /home/rstefanescu/r17_dataset/rev2/cyber-apk-nov2023-train-rev2/METADATA.csv --dictionary_path /home/rstefanescu/r17/scratch/result.json --pandas_path /home/rstefanescu/r17/scratch/output.csv

# Code capabilities

We implemented 4 different methods based on jacobians, discrete derivatives, Shapley values and model outputs. To aggregate the results we used cosine similarities of avg, avg of cos similarities, jensen-shannon, MSEavg, MAEavg, and adversarial_examples. We also provided three extra data augmentation options based on Drebbin dataset, Drebbin adversarial and a Poisoned dataset. The features from the potential poisoned model were tested against the features of a clean reference model. 

# Extra augmentation methods

## Drebbin dataset
This dataset is not given in the repository. TrojAI provided it in the form of four Numpy binary files. If these files are available, the path must be provided in the 
metaparameters variables infer_path_drebbin_x_train, infer_path_drebbin_x_test, infer_path_drebbin_y_train, and infer_path_drebbin_y_test. The x datasets have the size of (no_samples, 991) whereas the y datasets have the size of (no_samples,). 

Running the detector with this option is enabled by setting metaparameter infer_extra_data_augmentation to "drebinn".

## Drebin adversarial dataset 

To run with this option, you need to compute the adversarial examples for the Drebbin dataset for the reference model at models/id-00000001/. The adversarial examples calculation is enabled by setting the following metaparameters infer_load_drebbin and infer_calc_drebbin_adv to true. Also set the path where the adversarial examples will be saved in the metaparameter infer_path_adv_examples. The function calculating the adversarial examples is called in detector.py - infer_calc_drebbin_adv. The output consists of four np.ndarrays X_modified_class01_pc0.npy,  X_modified_class01_pc1.npy, X_modified_class10_pc0.npy, X_modified_class10_pc1.npy. It includes adversarial examples of size (no_samples, 991) that switch labels (from class 0 to 1 and class 1 to 0 denoted class01 or class10 in the files names) with respect to probability of class 0 or 1 (denoted pc0 or pc1 in the files names).  

Running the detector with this option is enabled by setting metaparameter infer_extra_data_augmentation to "drebinn_adversarial".

## Poison dataset
The path to poisoned examples is defined by command argument --examples_dirpath, metaparameter infer_path_poisoned_examples and poisoned_features.npy which is hardwired in get_poison_probability function defined in detector.py. 

Running the detector with this option is enabled by setting metaparameter infer_extra_data_augmentation to "poison".

# Experimental Results
## Jacobian Similarity
We compute the Jacobian of the reference model and test model at a set of data points. We then compute the cosine similarity of the Jacobians at each of those points. We then averager over the set of points and turn into a probability. 
### Real data
We use all provided sample data from the reference model and the test model. There are only 3 samples included with the reference model. Test models may have the same data.
- Results: 0.41764 AUC
### Random Boolean data
We generate 1000 random samples, uniform over [0,1]^991.
- Results: 0.47833 AUC

We also tried combining this with real data, no results of interest.
### Perturbed real data
We copy the real samples a few hunded times, then randomly swap 1% of features.
- Results: 0.50778 AUC
### cosine between average jacobian
In previous experiments, I computed cosines between the testa and ref jacobians on different data samples, then averaged.  In this experiment, I avergaged the jacobians first, then took the cosine.
- Results: 0.48403 AUC (random Boolean data)
- Results:  0.83972 AUC (perturbed real data) (NOT 0.6)

We copy 852 of the real testing samples a few times, then randomly swap 1% of features.
- Results: 0.79 AUC

We copy all the real testing samples a few times, then randomly swap 1% of features.
- Results: 0.79111 AUC

We copy all the real training samples a few times, then randomly swap 1% of features.
- Results: 0.79139 AUC

We use all the real training samples.
- Results: 0.49472 AUC

We use all the real testing samples.
- Results: 0.50111 AUC
### Experiments with p
We copy all the real training samples 5 times, then randomly swap p*100% of features.

- p=0.002: Results: 0.75417 AUC
- p=0.005: Results: 0.77 AUC
- p=0.01: Results: 0.79139 AUC
- p=0.02: Results: 0.80694 AUC
- p=0.03: Results: 0.80028 AUC
- p=0.05: Results: 0.77833 AUC
- p=0.1: Results: 0.31778 AUC (!)

### cosine output
I put in perturbed real data, computed the outputs, then computed the cosine between the output vectors
- Results: 0.69292 AUC
- Results with different output scaling: 0.66472 AUC (this change shouldn't affect AUC, so disrepancy must be from sample noise)
### evil config
Since the average cosine jacobian with real data was so bad, I flipped the sign. This is evil.
- Results: 0.54722

