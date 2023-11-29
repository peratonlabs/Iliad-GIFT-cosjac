This repo is adapted from [NIST's Round 17 example code](https://github.com/usnistgov/trojai-example/tree/cyber-apk-nov2023). 


# Setup the Conda environment

1. `conda create --name r17 python=3.8 -y`
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
sudo singularity build --force ./cyber-apk-nov2023_sts_cosjac.simg example_trojan_detector.def
```

# Container usage: Inferencing Mode

```
singularity run --nv ./cyber-apk-nov2023_sts_cosjac.simg infer --model_filepath ./models/id-00000001/model.pt --result_filepath ./scratch/result.txt --scratch_dirpath ./scratch --examples_dirpath ./models/id-00000001/clean-example-data --metaparameters_filepath ./metaparameters.json --schema_filepath ./metaparameters_schema.json --round_training_dataset_dirpath ./ --learned_parameters_dirpath ./learned_parameters
```


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

