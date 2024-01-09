import os
import torch
import numpy as np
import csv
import json
from torch.nn import Sequential
from copy import deepcopy

import shap
from utils.drebinnn import DrebinNN

# from utils.models import load_model, load_models_dirpath, ImageACModel, ResNetACModel
# from sklearn.ensemble import RandomForestRegressor

# from utils.abstract import AbstractDetector
#from utils.model_utils import compute_action_from_trojai_rl_model
#from utils.models import load_model, load_models_dirpath, ImageACModel, ResNetACModel


#from utils.world import RandomLavaWorldEnv
#from utils.wrappers import ObsEnvWrapper, TensorWrapper


def read_truthfile(truth_fn):
    with open(truth_fn) as f:
        truth = json.load(f)
    lc_truth = {k.lower(): v for k,v in truth.items()}
    return lc_truth

def read_number_from_csv(file_path):
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # Read the first row, which should contain the number
            for row in reader:
                number = int(row[0])  
                return number
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except IndexError:
        print(f"Invalid file format or empty file: {file_path}")
    except ValueError:
        print(f"Unable to convert the value to a number in file: {file_path}")

def cossim(v1,v2, eps=1e-8):
    #print(v1, v2)

    v1 = v1/(np.linalg.norm(v1) + eps)
    v2 = v2/(np.linalg.norm(v2) + eps)
    
    sim = (v1*v2).sum()

    return sim





def get_class_r14(truth_fn):
    cls = read_number_from_csv(truth_fn)
    return cls


def get_quants(x, n):
    """
    :param x:
    :param n:
    :return:
    """
    q = np.linspace(0, 1, n)
    return np.quantile(x, q)

def get_shapley_values(model: DrebinNN, 
                       dataset: list, 
                       test_dataset: list) -> list:
    '''
    Approximate Shapley values for the deep neural network model using the 
    backround dataset
    Input: model - deep neural network model
           dataset - backround dataset
           test_dataset - shapley values will be computed for the test dataset
    Output: list of torch tensors with shapley values
    '''
    shap.initjs()
    explainer_model = shap.GradientExplainer(model, dataset)
    shap_values_model = explainer_model.shap_values(test_dataset)
    return shap_values_model


def get_jac(model: DrebinNN, obs ):

    obs.requires_grad = True
    output = model(obs)
    jacobian = []
    y_onehot = torch.zeros([obs.shape[0], output.shape[1]], dtype=torch.long).to(obs.device)
    one = torch.ones([1], dtype=torch.long)
    for label in range(output.shape[1]):
        y_onehot.zero_()
        y_onehot[:, label] = one
        # curr_y_onehot = torch.reshape(y_onehot, out_shape)
        output.backward(y_onehot, retain_graph=True)
        jacobian.append(deepcopy(obs.grad.detach().cpu().numpy()))
        obs.grad.data.zero_()
    
    
    jac = np.stack(jacobian, axis=2)
    #avg_jac = np.mean(np.stack(jacobian, axis=0), axis=(1,2,3,4))
    return jac

def get_env_data(model, obs, env):
    done = False
    max_iters = 1000
    iters = 0
    reward = 0
    collect_jac = []
    while not done and iters < max_iters:
        #print(iters)
        jac = get_jac(model, obs)
        #print(jac.mean())
        collect_jac.append(jac)
        # print(f"Avg jac {jac}")
        env.render()
        action = compute_action_from_trojai_rl_model(model, obs, sample=True)
        # breakpoint()
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs,env)
        iters += 1
        
        done = terminated or truncated
    collect_jac = np.stack(collect_jac)
    mean_jac = collect_jac.mean(axis=0)
    # print(mean_jac)
    return mean_jac


def dissimilarity_detector(vectorFixed, Newvector):
    # Calculate Euclidean distance
    distance = np.linalg.norm(vectorFixed - Newvector)
    # Normalize the distance to [0, 1] range
    dissimilarity_score = distance / np.sqrt(len(vectorFixed))
    return dissimilarity_score


def get_grad_score(model_filepath,basepath):
    reference_model_path = os.path.join(basepath, "reference", "base_model", "model.pt")
    # load the model
    model, model_repr, model_class = load_model(model_filepath)
    # load ref model
    ref_model, ref_model_repr, ref_model_class = load_model(reference_model_path)
    # Load the config file
    config_dict = {}
    model_dirpath = os.path.dirname(model_filepath)
    #config_filepath = os.path.join(model_dirpath, 'config.json')
    config_filepath = os.path.join(model_dirpath, 'reduced-config.json')

    with open(config_filepath) as config_file:
        config_dict = json.load(config_file)
    
    size = config_dict["grid_size"]
    #print(size)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    model.eval()
    ref_model.eval()
    
    #model.train()
    #ref_model.train()

    # logging.info("Using compute device: {}".format(device))

    model_name = type(model).__name__
    observation_mode = "rgb" if model_name in [ImageACModel.__name__, ResNetACModel.__name__] else 'simple'

    wrapper_obs_mode = 'simple_rgb' if observation_mode == 'rgb' else 'simple'

    env = TensorWrapper(ObsEnvWrapper(RandomLavaWorldEnv(mode=observation_mode, grid_size=size), mode=wrapper_obs_mode))
    
    #obs, info = env.reset()
    #new_feat, ref_feat, sim = run_ep_2x(model, ref_model, obs, env)
    
    
    sims=[]
    
    for i in range(100):
    	obs, info = env.reset()
    	_,_,sim=run_ep_2x(model, ref_model, obs, env)
    	sims.append(sim)
    
    sim = np.mean(sims)
    
    

    #ref_feat = get_env_data(ref_model, obs, env)
    #obs, info = env.reset()
    #new_feat = get_env_data(model, obs, env)
    
    
    
    #feats_score = dissimilarity_detector(vectorFixed=ref_feat, Newvector=new_feat)
    #return feats_score
    return sim
    
    
    
def run_ep_2x(model1, model2, obs, env):
    done = False
    max_iters = 1000
    iters = 0
    reward = 0
    collect_jac1 = []
    collect_jac2 = []
    sims = [0]
    
    while not done and iters < max_iters:
        #print(iters)
        
        jac1 = get_jac(model1, obs)
        jac2 = get_jac(model2, obs)
        
        
        sim = cossim(jac1,jac2) #~0.78 AUC
        
        #print(jac1, jac2)
        
        #jac = get_jac(model, obs)
        #jac = get_jac(model, obs)
        
        collect_jac1.append(jac1)
        collect_jac2.append(jac2)
        

        
        
        logits1 = model1(obs)
        logits2 = model2(obs)
        
        #print(logits1.shape)
        probs1 = torch.softmax(logits1,1)
        probs2 = torch.softmax(logits2,1)
        
        sim_logits = cossim(logits1.detach().cpu().numpy(),logits2.detach().cpu().numpy())
        sim_probs = cossim(probs1.detach().cpu().numpy(),probs2.detach().cpu().numpy())
        
        
        #sim=sim_logits + sim
        sim=sim_logits    #     0.4192841490138787 AUC
        #sim=sim_probs    #     0.351990504017531 AUC
        
        if ~np.isnan(sim):
            sims.append(sim)
        
        
        action = compute_action_from_trojai_rl_model(model1, obs, sample=True)
        
        

        # print(f"Avg jac {jac}")
        #env.render()
        action = compute_action_from_trojai_rl_model(model1, obs, sample=True)
        # breakpoint()
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs,env)
        iters += 1
        done = terminated or truncated
    collect_jac1 = np.stack(collect_jac1)
    collect_jac2 = np.stack(collect_jac2)
    mean_jac1 = collect_jac1.mean(axis=0)
    mean_jac2 = collect_jac2.mean(axis=0)
    mean_sim = np.mean(sims)
    # print(mean_jac)
    return mean_jac1, mean_jac2, mean_sim
    
    
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def verify_binary_classifier(predictions, labels):
    """
    Function to compute precision, recall, F1 score, and ROC-AUC for a binary classifier.
    
    Args:
    predictions (list or array): Predicted probabilities or binary predictions.
    labels (list or array): True binary labels.
    
    Returns:
    dict: A dictionary containing precision, recall, F1 score, and ROC-AUC.
    """
    # Convert predictions to binary format (if they are not already binary)
    binary_predictions = [1 if p >= 0.5 else 0 for p in predictions]

    # Calculate precision, recall, and F1 score
    precision = precision_score(labels, binary_predictions)
    recall = recall_score(labels, binary_predictions)
    f1 = f1_score(labels, binary_predictions)

    # Calculate ROC-AUC
    # Note: roc_auc_score can handle both binary and continuous predictions
    roc_auc = roc_auc_score(labels, predictions)

    # Compile and return the results
    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    return results
    
