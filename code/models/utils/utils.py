from datetime import datetime
import io
import json
import gcsfs
from google.cloud import storage
from pathlib import Path
import pandas as pd
import pickle
import os
import yaml
from scikitplot.metrics import plot_lift_curve
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

def undersampling_v0(X_train, y_train, SEED):
    under_sampler = RandomUnderSampler(sampling_strategy='majority',
                                       random_state=SEED)

    sampled_x, sampled_y = under_sampler.fit_resample(X_train, y_train.astype(int))                                
    i = np.arange(len(sampled_y))
    np.random.shuffle(i)
    X_train = sampled_x.iloc[i]
    y_train = sampled_y.iloc[i]

    return X_train, y_train

def undersampling_v1(X_train, y_train, seed):
    under_sampler = RandomUnderSampler(sampling_strategy=0.115,
                                       random_state=seed)

    sampled_x, sampled_y = under_sampler.fit_resample(X_train, y_train.astype(int))                                
    i = np.arange(len(sampled_y))
    np.random.shuffle(i)
    X_train = sampled_x.iloc[i]
    y_train = sampled_y.iloc[i]

    return X_train, y_train

def undersampling_v2(X_train, y_train, seed):
    under_sampler = RandomUnderSampler(sampling_strategy=0.25,
                                       random_state=seed)

    sampled_x, sampled_y = under_sampler.fit_resample(X_train, y_train.astype(int))                                
    i = np.arange(len(sampled_y))
    np.random.shuffle(i)
    X_train = sampled_x.iloc[i]
    y_train = sampled_y.iloc[i]

    return X_train, y_train

def undersampling_v3(X_train, y_train, seed):
    under_sampler = RandomUnderSampler(sampling_strategy=0.44,
                                       random_state=seed)

    sampled_x, sampled_y = under_sampler.fit_resample(X_train, y_train.astype(int))                                
    i = np.arange(len(sampled_y))
    np.random.shuffle(i)
    X_train = sampled_x.iloc[i]
    y_train = sampled_y.iloc[i]

    return X_train, y_train

def no_undersampling(X_train, y_train, seed):
    return X_train, y_train

def get_undersampling(version):
    if version == 0:
        return undersampling_v0
    elif version == 1:
        return undersampling_v1
    elif version == 2:
        return undersampling_v2
    elif version == 3:
        return undersampling_v3
    else:
        return no_undersampling

def lift_curve(y_pred_prob, y_valid, step = 0.1):
    print('####################### Lift score curve #######################')
    y_probas = np.zeros((len(y_pred_prob), 2))
    for i, p in enumerate(y_pred_prob):
        y_probas[i,] = [1-p,p]
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_lift_curve(y_valid,y_probas, ax=ax)
    ax.legend(loc='upper right')
    # plt.savefig(os.path.join(lgbm_path,'figures/LGBM_lift_curve.png'))
    print('Computing lift score values - Step={}'.format(step))
    aux_lift = pd.DataFrame()
    aux_lift['real'] = y_valid
    aux_lift['predicted'] = y_pred_prob
    aux_lift.sort_values('predicted',ascending=False,inplace=True)
    x_val = np.arange(step,1+step,step)
    #Calculate the ratio of ones in validation set
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    y_val = []
    #Calculate for each x value its correspondent lift score value
    for x in x_val:
        num_data = int(np.ceil(x*len(aux_lift))) #ceil function returns the closest integer bigger than our number 
        data_here = aux_lift.iloc[:num_data,:]
        ratio_ones_here = data_here['real'].sum()/len(data_here)
        y_val.append(ratio_ones_here/ratio_ones)
    lift_frame = pd.DataFrame()
    lift_frame['percentile'] = x_val
    lift_frame['lift_score'] = y_val
    print(lift_frame)

def make_model_folders(
        root_path : Path, 
        output_folder_suffix: str = "generic"
    ):
    output_root_folder = str(root_path / "_".join(["output", output_folder_suffix]))
    run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S") 
    output_folder = output_root_folder+"/"+run_timestamp
    try:
        os.mkdir(output_folder)
    except FileNotFoundError:
        os.mkdir(output_root_folder)
        os.mkdir(output_folder)
    metadata_folder = output_root_folder+"/metadata"
    if not os.path.exists(metadata_folder):
        os.mkdir(metadata_folder)
    dataset_folder = output_root_folder+"/dataset"
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    return output_folder, metadata_folder, dataset_folder

def read_sql_file(file_path):
    return open(file_path, mode='r', encoding='utf-8-sig').read()

def read_csv_file(project_id, gcs_bucket, filename, cache_path=None, columns=None):
    output_dataframe = None
    if cache_path:
        local_filename = os.path.basename(filename)
        try:
            output_dataframe = pd.read_csv(cache_path+"/"+local_filename, usecols=columns, keep_default_na=False)
        except FileNotFoundError:
            # Download the file from gcs to cache_path
            storage_client = storage.Client(project_id)
            bucket = storage_client.get_bucket(gcs_bucket)
            blob = bucket.blob(filename)
            blob.download_to_filename(cache_path+"/"+local_filename)
            output_dataframe = pd.read_csv(cache_path+"/"+local_filename, usecols=columns, keep_default_na=False)
    else:
        output_dataframe = pd.read_csv("gs://"+gcs_bucket+"/"+filename, usecols=columns, keep_default_na=False)
    return output_dataframe

def read_csv_files_with_wildcard(project_id, gcs_bucket, filename_pattern, cache_path=None, columns=None):
    filenames = gcsfs.GCSFileSystem().glob("gs://"+gcs_bucket+"/"+filename_pattern)
    dataframes = [read_csv_file(project_id, 
                                gcs_bucket, 
                                filename[len(gcs_bucket)+1:], #removing bucket name from filename
                                cache_path,
                                columns=columns) for filename in filenames]
    return pd.concat(dataframes)

def read_yaml(path_to_yaml_file):
    config = None
    with open(path_to_yaml_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise(exc)
    return config

def save_yaml(input_dictionary, path_to_yaml_file):
    with open(path_to_yaml_file, 'w') as output_file:
        yaml.dump(input_dictionary, output_file)

def read_pickle(input_filename):
    with open(input_filename, 'rb') as filename:
        loaded_object = pickle.load(filename) 
    return loaded_object

def save_pickle(input_object, output_filename):
    with open(output_filename, 'wb') as filename:
        pickle.dump(input_object, filename)

def save_json(json_data, filepath):
    with open(filepath, 'w') as j:
        json.dump(json_data, j, indent=4)
        j.close()

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        f.close()
    return data

def save_params(dataset_params, 
                network_params, 
                training_params,
                folder_path,
                scaler= None):
    print(f'Saving params at {folder_path}')

    filepath = f'{folder_path}dataset.json'
    dataset_params = {key: str(value) for key, value in dataset_params.items()}
    save_json(dataset_params, filepath)

    filepath = f'{folder_path}network.json'
    network_params = {key: str(value) for key, value in network_params.items()}
    save_json(network_params, filepath)

    filepath = f'{folder_path}training.json'
    training_params = {key: str(value) for key, value in training_params.items()}
    save_json(training_params, filepath)

    if scaler:
        filepath = f'{folder_path}scaler.pkl'
        save_pickle(scaler, filepath)

def load_params(folder_path):
    output = {}
    for x in ['dataset', 'network', 'training']:
        filepath = f'{folder_path}{x}.json'
        output[x] = load_json(filepath)
    
    scaler_path = f'{folder_path}scaler.pkl'
    if os.path.exists(scaler_path):
        output['scaler'] = read_pickle(scaler_path)

    return output
