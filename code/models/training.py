import argparse
import logging
import os
import pickle
import pickle as pkl
import random
import re
import time
from datetime import datetime, timedelta
from os.path import dirname
from pathlib import Path

import joblib
import lightgbm as lgbm
import matplotlib
import numpy
import pandas
import seaborn
import shap
import yaml
from matplotlib import pyplot
from utils.bigQueryClient import BigQueryClient
from utils.google_utilities import *
from utils.HpTuningDynamicArgParser import HpTuningDynamicArgParser
from utils.ML_functions import *
from utils.pandas_utils import (memory_reduction,
                                       process_config_converters_dict)
from utils.plot_functions import *
from utils.utils import *
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import (SCORERS, accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, fbeta_score,
                             make_scorer, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     train_test_split)
from skopt import BayesSearchCV


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--project_id',
        help='Current Project id',
        required=True,
    )
    parser.add_argument(
        '--gcs_bucket',
        help='the GCS bucket used for dataset and metadata',
        required=True,
    )
    parser.add_argument(
        '--dataset_filename_wildcard',
        help='wildcard path for the dataset in the GCS bucket, without gs://bucket_id',
        required=True,
    )
    parser.add_argument(
        '--env',
        help='Environment (dev,test,prod) for exporting model and other metadata to output bucket',
        required=True,
    )
    parser.add_argument(
        '--model_name',
        help='Model name: it can be either cinema, sport or calcio',
        required=True,
    )
    parser.add_argument(
        '--model_version',
        help='Model version for exporting model and other metadata to output bucket',
        required=True,
    )
    parser.add_argument(
        '--drop_autocorrelated',
        help='Set to True if the drop_autocorrelated feature of the preprocessing is wanted',
        default=False,
        required=False,
    )
    parser.add_argument(
        '--hp_tune',
        help='Hyper parameter tuning mode: if specified it can only be gridsearch',
        default=False,
        required=False,
    )
    return vars(parser.parse_args())

def init_folders(job_args, config, root_path):

    # make local folders
    output_folder, metadata_folder, dataset_folder = make_model_folders(root_path,
                                                                        output_folder_suffix=job_args["model_name"])

    # metadata folder in GCS
    gcs_metadata_path = config["use_case_id"] + "/" + job_args["env"] + "/model/" + \
                        job_args["model_version"]

    # config file to output_folder and gcs
    save_yaml(config, output_folder + "/training_config.yaml")
    upload_file_to_gcs(job_args["gcs_bucket"], output_folder + "/training_config.yaml",
                       gcs_metadata_path + '/training_config.yaml')
    save_yaml(job_args, output_folder + "/job_args.yaml")
    upload_file_to_gcs(job_args["gcs_bucket"], output_folder + "/job_args.yaml", gcs_metadata_path + '/job_args.yaml')

    return output_folder, metadata_folder, dataset_folder, gcs_metadata_path


def data_preprocessing(job_args, 
                       config, 
                       metadata_folder, 
                       dataset_folder, 
                       gcs_metadata_path,
                       logger):

    #################################
    ########## DATA IMPORT ##########
    #################################

    data = read_csv_files_with_wildcard(
        project_id=job_args["project_id"],
        gcs_bucket=job_args["gcs_bucket"],
        filename_pattern=job_args["dataset_filename_wildcard"],
        # cache_path=dataset_folder,

        columns = list(set(config["DATASET"]['columns_to_keep'] + config["DATASET"]["columns_to_filter_or_preprocess"] + [
                  config["DATASET"]["index_column"], config["DATASET"]["target_column"]])),
        # converters=process_config_converters_dict(config["DATASET"]["converters"])
    )

    data = memory_reduction(data)
    data = data.set_index(config["DATASET"]["index_column"])

    logger.info("Successfully downloaded dataset")
    logger.info("dataset lenght: {}".format(len(data)))
    logger.info("n columns: {}".format(len(data.columns)))
    logger.debug("Dataset dtypes: {}".format(data.dtypes))

    
    ### Target distribution
    logger.info('Distribution of data by flag target:')
    logger.info("{}".format(data[config["DATASET"]["target_column"]].value_counts()))

    ### Null values
    null_values = data.isna().sum().to_frame('Nulls')
    logger.info('List of variables with null values:')
    logger.info("{}".format(null_values[null_values['Nulls'] > 0]))

    ### Columns to keep
    target_col = data[config["DATASET"]["target_column"]]
    features = data[config["DATASET"]['columns_to_keep']]

    logger.info('List of variables included in the model:')
    logger.info(str(features.columns.to_list()))

    ######################################
    ########### PREPROCESSING ############
    ######################################

    features = features.rename(columns={x: f'cat_{x}' for x in features.select_dtypes(include='object').columns})

    ### Categorical conversion, One Hot Encoding, Drop low std features
    dpp = NewDataPreProcess(
        features,
        etl_type='train',
        freq_treshold=config["PREPROCESS"]['freq_treshold'],
        corr_treshold=config["PREPROCESS"]['corr_treshold'],
        drop_autocorr=bool(job_args["drop_autocorrelated"]),
        pkl_name=metadata_folder + '/' + config["PREPROCESS"]['filename'],
        target_col=target_col,
        logger=logger
    )
    x_all = dpp.prepare_data()

    # upload preprocess pickle
    upload_file_to_gcs(job_args["gcs_bucket"], metadata_folder + '/' + config["PREPROCESS"]['filename'],
                       gcs_metadata_path + '/' + config["PREPROCESS"]['filename'])

    ### Modify columns name
    x_all.columns = [re.sub(' ', '_', x) for x in x_all.columns]
    x_all.columns = [re.sub('cat_', '', x) for x in x_all.columns]

    return x_all, target_col


def training(data, 
             target_col, 
             config, 
             metadata_folder,
             output_folder,
             gcs_metadata_path,
             logger,
             job_args):

    ### Split Training vs Test
    x_train, x_test, y_train, y_test = train_test_split(data, target_col,
                                                        test_size=config["MODEL"]["settings"]["test_size"],
                                                        random_state=config["MODEL"]["settings"]["random_state"],
                                                        stratify=target_col)

    logger.info('x_train has {} records'.format(x_train.shape[0]))
    logger.info('y_train has {} records'.format(y_train.shape[0]))
    logger.info('x_test has {} records'.format(x_test.shape[0]))
    logger.info('y_test has {} records'.format(y_test.shape[0]))

    # Undersampling
    undersampling_fn = get_undersampling(config["MODEL"]["settings"]["undersampling_version"])

    x_train_res, y_train_res = undersampling_fn(x_train, y_train, config['MODEL']['settings']['random_state'])
    logger.info(f'x_train undersampled shape: {x_train_res.shape}')
    logger.info(f'y_train undersampled shape: {y_train_res.shape}')
    #################################
    ######### MODEL FITTING #########
    #################################

    if job_args["hp_tune"] == "bayes":

        ### Cross Validation with GridSearch ###
        logger.info("Model fitting with Cross Validation")

        base_clf = lgbm.LGBMClassifier()
        # Create a Bayesian optimization object
        bayes_search = BayesSearchCV(
                                    base_clf, 
                                    param_space= config["CV"]['bayes_settings']['param_space'], 
                                    n_iter=config["CV"]['bayes_settings']['n_iter'], 
                                    cv=config["CV"]['bayes_settings']['cv'],
                                    n_jobs=config["CV"]['bayes_settings']['n_jobs'],
                                    scoring=config["CV"]['bayes_settings']['scoring'])

        # Perform hyperparameter tuning with Bayesian optimization
        bayes_search.fit(x_train_res, y_train_res,
                eval_set=[(x_test, y_test), (x_train_res, y_train_res)])

        best_parameters = bayes_search.best_params_
        print(best_parameters)
        save_yaml(best_parameters, metadata_folder + '/best_parameters.yaml')
        upload_file_to_gcs(job_args["gcs_bucket"], metadata_folder + '/best_parameters.yaml',
                           gcs_metadata_path + '/best_parameters.yaml')
        best_model = bayes_search.best_estimator_

        y_prob = best_model.predict_proba(x_test)[:, 1]
        ### filename of the model
        model_name = config['CV'][config['choosen_model']]['filename']
        ### Save/Load model output from gridsearch
        joblib.dump(clf_lgbm, metadata_folder + '/' + model_name)
        bst = joblib.load(metadata_folder + '/' + model_name)
        bst.booster_.save_model(metadata_folder + '/' +model_name_txt)
        upload_file_to_gcs(job_args["gcs_bucket"], metadata_folder + '/' + model_name, gcs_metadata_path + '/' + model_name)
        upload_file_to_gcs(job_args["gcs_bucket"], metadata_folder + '/' + model_name_txt, gcs_metadata_path + '/' + model_name_txt)
       

    else:
        ### No Cross Validation
        logger.info("Model fitting without Cross Validation")
        best_parameters = config['MODEL'][config['choosen_model']]['params']
        metric = best_parameters['metric']

    logger.info(best_parameters)
    clf_lgbm = lgbm.LGBMClassifier(**best_parameters)
    clf_lgbm.fit(x_train, y_train, eval_set=[(x_test, y_test), (x_train_res, y_train_res)], eval_names=['valid_0','train'], verbose=1)
    clf_lgbm_auc = clf_lgbm._best_score['valid_0'][metric]
    logger.info('Best score obtained is: {}'.format(clf_lgbm_auc))

    ### filename of the model
    model_name = config['MODEL'][config['choosen_model']]['filename']
    model_name_txt = config['MODEL'][config['choosen_model']]['filename_txt']

    ### Save/Load model
    joblib.dump(clf_lgbm, metadata_folder + '/' + model_name)
    bst = joblib.load(metadata_folder + '/' + model_name)
    bst.booster_.save_model(metadata_folder + '/' +model_name_txt)

    upload_file_to_gcs(job_args["gcs_bucket"], metadata_folder + '/' + model_name, gcs_metadata_path + '/' + model_name)
    upload_file_to_gcs(job_args["gcs_bucket"], metadata_folder + '/' + model_name_txt, gcs_metadata_path + '/' + model_name_txt)

    ### Classification report
    y_pred_train = clf_lgbm.predict(x_train_res)
    y_pred_test = clf_lgbm.predict(x_test)
    y_proba_test = clf_lgbm.predict_proba(x_test)

    ### Classification report
    cm_training = classification_report(y_train_res, y_pred_train)
    cm_test = classification_report(y_test, y_pred_test)

    logger.info('Classification report for the Training dataset:')
    logger.info(cm_training)
    logger.info('Classification report for the Test dataset:')
    logger.info(cm_test)

    ### Confusion matrix
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_test), columns=['pred_0', 'pred_1'],
                               index=['obs_0', 'obs_1'])
    logger.info('Confusion Matrix on Test dataset:')
    logger.info(conf_matrix.to_string())

    #######################################
    ######### LEARNING CURVES #############
    #######################################

    lgbm.plot_metric(clf_lgbm)
    pyplot.savefig(output_folder + "/learning_curves.png", dpi=200)
    upload_file_to_gcs(job_args["gcs_bucket"], output_folder + "/learning_curves.png",
                       gcs_metadata_path + "/learning_curves.png")

    #########################
    ######### LIFT ##########
    #########################

    serie_y_test = pd.Series(y_test, dtype='float', name="y_real").reset_index()
    serie_proba_pred = pd.Series(y_proba_test[:, 1], dtype='float', name="y_proba_pred").reset_index()
    new_df = pd.concat([serie_y_test, serie_proba_pred], axis=1)
    lift = model_uplift(df=new_df, tgt_var='y_real', score_var='y_proba_pred', logger=logger)
    logger.info(lift.to_string()) 
    lift.to_csv(output_folder+"/model_uplift.csv", index=False )
    upload_file_to_gcs(job_args["gcs_bucket"], output_folder + "/model_uplift.csv", gcs_metadata_path+'/model_uplift.csv')

    #######################################
    ######### FEATURE IMPORTANCE ##########
    #######################################

    num_feats = len(x_train.columns)
    ax = lgbm.plot_importance(clf_lgbm, importance_type=config['FEATURE_IMPORTANCE']['importance_type'],
                              figsize=(10, num_feats / 2.5))
    ax.figure.savefig(output_folder + '/' + config['FEATURE_IMPORTANCE']['filename'] + '.png')
    upload_file_to_gcs(job_args["gcs_bucket"], output_folder + '/' + config['FEATURE_IMPORTANCE']['filename'] + '.png',
                       gcs_metadata_path + '/' + config['FEATURE_IMPORTANCE']['filename'] + '.png')

    #######################################
    ######### SHAP VALUES #################
    #######################################
    ## for train
    shap_output_folder = output_folder + '/shap_graphs'
    type_shap=config['SHAP_VALUES']['type_shap']
    output_filename_prefix = 'TRAIN_' + config['SHAP_VALUES']['filename'] 
    treshold = config['SHAP_VALUES']['importance_treshold']

    logger.info(f"N shaps: {config['SHAP_VALUES']['n_shaps']}")

    plot_shap_values(
        input_dataframe=x_train,
        input_model=clf_lgbm,
        n_shaps=config['SHAP_VALUES']['n_shaps'],
        type_shap=type_shap,
        importance_treshold=config['SHAP_VALUES']['importance_treshold'],
        output_folder=shap_output_folder,
        output_filename_prefix=output_filename_prefix,
        logger=logger
    )
    ## for test
    type_shap=config['SHAP_VALUES']['type_shap']
    output_filename_prefix = 'TEST_' + config['SHAP_VALUES']['filename'] 
    treshold = config['SHAP_VALUES']['importance_treshold']
    plot_shap_values(
        input_dataframe=x_test,
        input_model=clf_lgbm,
        n_shaps=config['SHAP_VALUES']['n_shaps'],
        type_shap=type_shap,
        importance_treshold=config['SHAP_VALUES']['importance_treshold'],
        output_folder=shap_output_folder,
        output_filename_prefix=output_filename_prefix,
        logger=logger
    )
    ## upload all files to gcs
    upload_folder_to_gcs(job_args["gcs_bucket"], shap_output_folder, gcs_metadata_path)

    #######################################
    ######### ABS SHAP  ###################
    #######################################
    
    plot_abs_SHAP_summary(x_test,clf_lgbm,output_folder,max_display=40)
    upload_file_to_gcs(job_args["gcs_bucket"], output_folder+"/importance_shap_values.png", gcs_metadata_path+"/importance_shap_values.png")


if __name__ == "__main__":

    job_args = parse_arguments()
    logger = create_logger('ita_sky_{}'.format(job_args["model_name"]),
                                               job_args["project_id"])

    root_path = Path(__file__).parent
    config_path = root_path / 'config' / 'training'

    ######### Load config file #########
    config = read_yaml(config_path / "config_{}.yaml".format(job_args["model_name"]))
    
    ### Init Folders 
    output_folder, metadata_folder, dataset_folder, gcs_metadata_path = init_folders(job_args, config, root_path)

    logger.info(f'GCS metadata output folder: {gcs_metadata_path}')


    # Preprocessing
    data, target_col = data_preprocessing(job_args, 
                                          config, 
                                          metadata_folder, 
                                          dataset_folder, 
                                          gcs_metadata_path,
                                          logger)
    

    # Training
    training(data, 
             target_col, 
             config, 
             metadata_folder,
             output_folder,
             gcs_metadata_path,
             logger,
             job_args)

    