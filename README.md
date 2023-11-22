This repo is adapted from [NIST's Round 17 example code](https://github.com/usnistgov/trojai-example/tree/cyber-apk-nov2023). 


# Setup the Conda environment

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


# Build a new container 

```
sudo singularity build ./cyber-apk-nov2023_sts_cosjac.simg example_trojan_detector.def
```

# Container usage: Inferencing Mode

```
singularity run --nv ./cyber-apk-nov2023_sts_cosjac.simg infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters
```



