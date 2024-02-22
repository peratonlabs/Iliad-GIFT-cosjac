""" Entrypoint to interact with the detector.
"""
import json
import logging
import os
import warnings

from argparse import ArgumentParser
import jsonschema

from detector import Detector, Preprocess

warnings.filterwarnings("ignore")

def test_schema(metaparameters_filepath: str, scheme_filepath: str):
    '''
    Verify if metaparameters definitions satisfy the provided scheme
    Args:
        metaparameters_filepath: path location of the metaparameters file
        schema_filepath: path location of the metaparameters scheme file 
    '''
    with open(metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(scheme_filepath) as schema_file:
        schema_json = json.load(schema_file)

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)    

def inference_mode(args):
    
    # Validate config file against schema
    test_schema(args.metaparameters_filepath, args.schema_filepath)
    
    # Create the detector instance and loads the metaparameters.
    detector = Detector(
        args.metaparameters_filepath,
        args.learned_parameters_dirpath,
        args.reference_model_path
    )

    logging.info("Calling the trojan detector")
    detector.infer(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath, args.round_training_dataset_dirpath)

def configure_mode(args):
    # Validate config file against schema
    test_schema(args.metaparameters_filepath, args.schema_filepath)
    # Create the detector instance and loads the metaparameters.
    preproc = Preprocess(
        args.metaparameters_filepath,
        args.learned_parameters_dirpath,
        args.reference_model_path,
        args.reference_model_origin,
        args.drebbin_dataset_dirpath,
        args.poison_dataset_path
    )

    logging.info("Calling configuration mode")
    preproc.configure(
        args.reference_model_path, 
        args.automatic_configuration
    )

def add_infer_subparser_arguments(subparser):
    subparser.add_argument(
    "--model_filepath",
    type=str,
    help="File path to the pytorch model file to be evaluated.",
    required=True
    )

    subparser.add_argument(
    "--result_filepath",
    type=str,
    help="File path to the file where output result should be written. After "
    "execution this file should contain a single line with a single floating "
    "point trojan probability.",
    required=True
    )
    subparser.add_argument(
    "--scratch_dirpath",
    type=str,
    help="File path to the folder where scratch disk space exists. This folder will "
    "be empty at execution start and will be deleted at completion of "
    "execution.",
    required=True
    )
    subparser.add_argument(
    "--examples_dirpath",
    type=str,
    help="File path to the folder of examples which might be useful for determining "
    "whether a model is poisoned.",
    required=True
    )
    subparser.add_argument(
    "--round_training_dataset_dirpath",
    type=str,
    help="File path to the directory containing id-xxxxxxxx models of the current "
    "rounds training dataset.",
    required=True
    )
    subparser.add_argument(
    "--metaparameters_filepath",
    help="Path to JSON file containing values of tunable paramaters to be used "
    "when evaluating models.",
    type=str,
    required=True,
    )
    subparser.add_argument(
    "--schema_filepath",
    type=str,
    help="Path to a schema file in JSON Schema format against which to validate "
    "the config file.",
    required=True,
    )
    subparser.add_argument(
    "--learned_parameters_dirpath",
    type=str,
    help="Path to a directory containing parameter data (model weights, etc.) to "
    "be used when evaluating models.  If --configure_mode is set, these will "
    "instead be overwritten with the newly-configured parameters.",
    required=True,
    )
    subparser.add_argument(
    "--reference_model_origin",
    type=str,
    help="Path to the reference clean model source",
    default="models/id-00000001/",
    required=False,
    )
    subparser.add_argument(
    "--reference_model_path",
    type=str,
    help="Path to the reference clean model in the container",
    default="/learned_parameters/models/id-00000001",
    required=False,
    )
    
    subparser.add_argument("--source_dataset_dirpath", type=str)

def add_configure_subparser_arguments(subparser):

    subparser.add_argument(
        "--metaparameters_filepath",
        help="Path to JSON file containing values of tunable paramaters to be used "
        "when evaluating models.",
        type=str,
        required=True,
    )
    subparser.add_argument(
        "--schema_filepath",
        type=str,
        help="Path to a schema file in JSON Schema format against which to validate "
        "the config file.",
        required=True,
    )
    subparser.add_argument(
        "--learned_parameters_dirpath",
        type=str,
        help="Path to a directory containing parameter data (model weights, etc.) and "
        "dependencies.",
        required=True,
    )
    subparser.add_argument(
        '--automatic_configuration',
        help='Whether to enable automatic training or not, which will retrain the detector across multiple variables',
        action='store_true',
    )
    subparser.add_argument(
    "--reference_model_origin",
    type=str,
    help="Path to the reference clean model",
    default="models/id-00000001/",
    required=False,
    )
    subparser.add_argument(
    "--drebbin_dataset_dirpath",
    type=str,
    help="Local system full path to drebbin_dataset_dirpath",
    default='~/cyber-apk-nov2023-vectorized-drebin',
    required=False
    )
    subparser.add_argument(
    "--reference_model_path",
    type=str,
    help="Path to the reference clean model in the container",
    default="/learned_parameters/models/id-00000001",
    required=False,
    )
    subparser.add_argument(
    "--poison_dataset_path",
    type=str,
    help="Local system full path to drebbin_dataset_dirpath",
    default='~/poison_data',
    required=False
    )
if __name__ == "__main__":

    temp_parser = ArgumentParser(add_help=False)

    parser = ArgumentParser(
        description="Template Trojan Detector to Demonstrate Test and Evaluation. Should be customized to work with target round in TrojAI."
        "Infrastructure."
    )

    parser.set_defaults(func=lambda args: parser.print_help())
    subparser = parser.add_subparsers(dest='cmd', required=True)
    inf_parser = subparser.add_parser('infer', help='Execute container in inference mode for TrojAI detection.')
    add_infer_subparser_arguments(inf_parser)
    inf_parser.set_defaults(func=inference_mode)
    
    configure_parser = subparser.add_parser('configure', help='Execute container in configuration mode for TrojAI detection. This will produce a new set of learned parameters to be used in inference mode.')
    add_configure_subparser_arguments(configure_parser)
    configure_parser.set_defaults(func=configure_mode)

    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        )

    args, extras = temp_parser.parse_known_args()

    if '--help' in extras or '-h' in extras:
        args = parser.parse_args()
    # Checks if new mode of operation is being used, or is this legacy
    elif len(extras) > 0 and extras[0] in ['infer', 'configure', 'preprocess']:
        args = parser.parse_args()
        args.func(args)
    else:
        # Assumes we have inference mode if the subparser is not used
        args = inf_parser.parse_args()
        args.func(args)
