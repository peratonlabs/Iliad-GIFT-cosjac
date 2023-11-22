This repo is adapted from [NIST's Round 17 example code](https://github.com/usnistgov/trojai-example/tree/cyber-apk-nov2023). 


# Setup the Conda Environment

(haven't check this yet)

1. `conda create --name r17 python=3.9 -y`
2. `conda activate r17`
3. Install required packages into this conda environment
 
    - `conda install pytorch=1.12.1=py3.8_cuda10.2_cudnn7.6.5_0 -c pytorch`
    - `pip install tqdm jsonschema jsonargparse scikit-learn`


# Run inference outside of Singularity

```
python entrypoint.py infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters
```


# Build a New Container 

```
sudo singularity build ./containers/example_trojan_detector.simg example_trojan_detector.def
```




# Container usage: Inferencing Mode

(not updatad yet)

Example usage for inferencing:
   ```bash
   python entrypoint.py infer \
   --model_filepath <model_filepath> \
   --result_filepath <result_filepath> \
   --scratch_dirpath <scratch_dirpath> \
   --examples_dirpath <examples_dirpath> \
   --round_training_dataset_dirpath <round_training_dirpath> \
   --metaparameters_filepath <metaparameters_filepath> \
   --schema_filepath <schema_filepath> \
   --learned_parameters_dirpath <learned_params_dirpath>
   ```






