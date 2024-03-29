import argparse
import subprocess
import glob
import os
from sklearn.metrics import roc_auc_score
import pandas as pd
import json
import time
from concurrent.futures import ThreadPoolExecutor
from utils.utils import save_dictionary_to_file


def run_command(model_file: str, gpu_id: int) -> tuple:
    '''
    Run command via subprocess and return 
    process stdout, stderr and returncode
    Args:
        model_file - path where model is saved
        gpu_id - integer with the gpu_id
    Outputs:
        A tuple of strings with stdout, stderr
        and returncode
    '''

    path_components = model_file.split(os.sep)
    result_file = './scratch/' + path_components[-2] + '.json'

    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    command = (
        f"python entrypoint.py infer --model_filepath {model_file} "
        f"--result_filepath {result_file} --scratch_dirpath"
        " ./scratch --examples_dirpath ./models/id-00000001/clean-example-data"
        " --metaparameters_filepath ./metaparameters.json --schema_filepath "
        "./metaparameters_schema.json --round_training_dataset_dirpath "
        "./ --learned_parameters_dirpath ./learned_parameters"
    )

    process = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True
    )

    return process.stdout, process.stderr, process.returncode


# Create the parser
parser = argparse.ArgumentParser(
    description="Compute poison probability for all models locally!"
)
# Add optional arguments
parser.add_argument(
    "--test_models_path",
    type=str,
    help="Test models path",
    required=True
)

parser.add_argument(
    "--metadata_path",
    type=str,
    help="Test metadata path",
    required=True
)

parser.add_argument(
    "--dictionary_paths",
    type=str,
    help="paths for dictionary with models names and poison prob path",
    required=True
)

parser.add_argument(
    "--pandas_path",
    type=str,
    help="Pandas with poison labels and prob csv path",
    required=True
)

parser.add_argument(
    "--metaparameters_file_path",
    type=str,
    help="Metaparameters file path",
    required=True
)

# Parse the arguments
args = parser.parse_args()


start_time = time.time()
# Base directory
base_dir = args.test_models_path

# Pattern to match all 'model.pt' files in the subdirectories
pattern = os.path.join(base_dir, 'id*/model.pt')

# Find all files matching the pattern
model_files = glob.glob(pattern)
model_files = sorted(model_files)
no_available_GPUs = 2
# Using ThreadPoolExecutor to run commands in parallel
with ThreadPoolExecutor(max_workers=8) as executor:
    # Schedule the tasks, alternating GPU assignment
    futures = [executor.submit(run_command, model_file, inx % no_available_GPUs) for inx, model_file in enumerate(model_files)]

    for future in futures:
        output, error, exit_code = future.result()
        print(f"Output: {output}")
        print(f"Error: {error}")
        print(f"Exit Code: {exit_code}")

df = pd.read_csv(args.metadata_path)

metaparameters = json.load(open(args.metaparameters_file_path, "r"))

pattern = os.path.join(args.dictionary_paths, 'id*.json')

# Find all files matching the pattern
output_files = glob.glob(pattern)
output_files = sorted(output_files)

my_dict = {}
for file_path in output_files:
    with open(file_path, 'r') as file:
        existing_data = json.load(file)
    my_dict.update(existing_data)

output_file = (
    metaparameters['infer_feature_extraction_method'] +
    '_' + 
    metaparameters['infer_proximity_aggregation_method'] +
    '_' + 
    'result.json'
)

save_dictionary_to_file(
    my_dict,
    os.path.join(args.dictionary_paths, output_file)
)

for file_path in output_files:
    try:
        os.remove(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")

# Convert the dictionary to a list of tuples (items)
data_items = list(my_dict.items())

# Create DataFrame
df_prob = pd.DataFrame(data_items, columns=['model_name', 'probability'])
merged_df = pd.merge(df[['model_name', 'poisoned']], df_prob, on='model_name', how='inner')

# Convert 'poisoned' to integers (True to 1 and False to 0)
merged_df['poisoned'] = merged_df['poisoned'].astype(int)

# Apply roc_auc_score
auc_score = roc_auc_score(merged_df['poisoned'], merged_df['probability'])
print(f'ROC AUC Score: {auc_score}')

output_file = (
    metaparameters['infer_feature_extraction_method'] +
    '_' + 
    metaparameters['infer_proximity_aggregation_method'] +
    '_' + 
    'roc_auc.json'
)

save_dictionary_to_file(
    {'auc_score': auc_score},
    os.path.join(args.dictionary_paths, output_file)
)

output_file = (
    metaparameters['infer_feature_extraction_method'] +
    '_' + 
    metaparameters['infer_proximity_aggregation_method'] +
    '_' + 
    'output.csv'
)

merged_df.to_csv(os.path.join(args.pandas_path, output_file), index=False)
end_time = time.time()
# Calculate the runtime
runtime = end_time - start_time

print(f"The runtime is {runtime} seconds.")
