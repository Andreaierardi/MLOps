import argparse
import re
import warnings
from pathlib import Path

import joblib
from utils.bigQueryClient import BigQueryClient
from utils.google_utilities import *
from utils.google_utilities import create_logger, download_blob_file
from utils.ML_functions import *
from utils.plot_functions import *
from utils.utils import *

warnings.simplefilter('ignore')

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
        '--date_prediction_reference',
        help='Date of reference of the kpis and for generating predictions',
        required=True)

    parser.add_argument(
        '--date_prediction_run',
        help='Date of reference of the kpis and for generating predictions',
        required=True)

    parser.add_argument(
        '--export_project_id',
        help='Project ID where to export results',
        required=True)

    parser.add_argument(
        '--export_dataset_id',
        help='Dataset ID where to export results',
        required=True)

    parser.add_argument(
        '--export_table_id',
        help='Table ID where to export results',
        required=True)

    return vars(parser.parse_args())

def init_folders(job_args, config, root_path):

    # make local folders
    output_folder, metadata_folder, dataset_folder = make_model_folders(root_path,
                                                                        output_folder_suffix=job_args["model_name"])

    # metadata folder in GCS
    gcs_metadata_path = config["use_case_id"] + "/" + job_args["env"] + "/model/" + \
                        job_args["model_version"]

    return output_folder, metadata_folder, dataset_folder, gcs_metadata_path

def data_preprocessing(job_args,
                       config,
                       training_config,
                       preprocess_config,
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
            columns= list(set(training_config["DATASET"]['columns_to_keep'] +
                        training_config["DATASET"]["columns_to_filter_or_preprocess"] + 
                        # ['flg_target'] + #---> DA RIMUOVERE <----
                        [config["dataset_index_column_name"], config["kpis_run_date_column_name"], 
                        ] + [config['general_profiling_consent_column_name']]))                
                    )

    logger.info("Successfully downloaded dataset")
    logger.info("dataset length: {}".format(len(data)))
    logger.info("n columns: {}".format(len(data.columns)))
    logger.debug("Dataset dtypes: {}".format(data.dtypes))

    ### Null values
    null_values = data.isna().sum().to_frame('Nulls')
    logger.info('List of variables with null values:')
    logger.info(null_values[null_values['Nulls'] > 0].to_string())
    
    data = data.set_index(config["dataset_index_column_name"])

    consent_col = data[config['general_profiling_consent_column_name']]
    data = data.drop(config['general_profiling_consent_column_name'], axis=1)
    date_kpis_run = data[config["kpis_run_date_column_name"]].iloc[0]

    logger.info('List of variables included in the model:')
    logger.info(str(data.columns))
    logger.info('PREDICTION START')

    # download preprocess.pickle from gcs
    download_blob_file(project_id=job_args["project_id"],
                       bucket_name=job_args["gcs_bucket"],
                       source_blob_name=gcs_metadata_path + '/' + preprocess_config['filename'],
                       local_blob_name=metadata_folder + '/' + preprocess_config['filename'])

    # download ups_cinema_lgbm.pkl from gcs
    download_blob_file(project_id=job_args["project_id"],
                       bucket_name=job_args["gcs_bucket"],
                       source_blob_name=gcs_metadata_path + '/' + training_config['MODEL'][training_config['choosen_model']]['filename'],
                       local_blob_name=metadata_folder + '/' +
                                       training_config['MODEL'][training_config['choosen_model']]['filename'])

    download_blob_file(project_id=job_args["project_id"],
                       bucket_name=job_args["gcs_bucket"],
                       source_blob_name=gcs_metadata_path + '/' + training_config['MODEL'][training_config['choosen_model']]['filename_txt'],
                       local_blob_name=metadata_folder + '/' + training_config['MODEL'][training_config['choosen_model']]['filename_txt'])

    ### Categorical conversion, One Hot Encoding, Drop low std features
    data = data.rename(columns={x: f'cat_{x}' for x in data.select_dtypes(include='object').columns})

    dpp = NewDataPreProcess(data, etl_type ='pred',
                            freq_treshold=preprocess_config['freq_treshold'],
                            corr_treshold=preprocess_config['corr_treshold'],
                            pkl_name=metadata_folder+'/'+preprocess_config['filename'],
                            logger=logger)
    x_all = dpp.prepare_data()

    ### Modify columns name
    x_all.columns = [re.sub(' ', '_' ,x) for x in x_all.columns]
    x_all.columns = [re.sub('cat_', '' ,x) for x in x_all.columns]

    return x_all, consent_col, date_kpis_run

def prediction(data,
               consent_col, 
               date_kpis_run,
               config,
               training_config,
               metadata_folder,
               logger,
               job_args
               ):
    pickle_model = joblib.load(open(metadata_folder + '/' +
                                    training_config['MODEL'][training_config['choosen_model']]['filename'], 'rb'))


    bst_pred = lgbm.Booster(model_file=metadata_folder + '/' + training_config['MODEL'][training_config['choosen_model']]['filename_txt'])
    cols = bst_pred.feature_name()

    # sorting column based on prediction
    data_final = data[list(data[cols].columns)]
    data_final = data_final.reindex(columns=cols)

    ### Prediction
    Ypredict = pickle_model.predict_proba(data_final)
    prob = pd.Series(Ypredict[:, 1], dtype='float', name=config["score_column_name"])
    
    result_df = pd.DataFrame()
    result_df[config["dataset_index_column_name"]] = data.index
    result_df[config["score_column_name"]] = prob
    result_df[config["decile_column_name"]] = pd.qcut(result_df[config["score_column_name"]], 10,
                                                        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    result_df[config["model_version_column_name"]] = job_args["model_version"]
    result_df[config["kpis_run_date_column_name"]] = date_kpis_run
    result_df[config["run_date_column_name"]] = job_args["date_prediction_run"]
    result_df[config["prediction_reference_date_column_name"]] = job_args["date_prediction_reference"]
    result_df = result_df.merge(consent_col, how='left', left_on='contract_code', right_index= True)

    logger.info("result_df shape: {}".format(result_df.shape))

    # computing the decile of the consented CB
    result_df_consented = result_df[result_df[config["general_profiling_consent_column_name"]]=="Y"]
    result_df_consented[config["decile_consented_column_name"]] = pd.qcut(result_df_consented[config["score_column_name"]], 10,
                                  labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    result_df_consented.set_index(config["dataset_index_column_name"], inplace=True)                                  
    result_df_consented = result_df_consented[[config["decile_consented_column_name"]]]
    logger.info("result_df_consented shape: {}".format(result_df_consented.shape))
    
    # joining the Full CB with the Consented CB for the export
    result_df = result_df.join(result_df_consented, how="left", on=config["dataset_index_column_name"])
    logger.info("joined result_df shape: {}".format(result_df.shape))


    # Setting Big Query Client
    logger.info("Final dataset export")
    output_schema = [
                bigquery.SchemaField(config["dataset_index_column_name"], bigquery.enums.SqlTypeNames.STRING),				
                bigquery.SchemaField(config["general_profiling_consent_column_name"], bigquery.enums.SqlTypeNames.STRING),				
                bigquery.SchemaField(config["score_column_name"], bigquery.enums.SqlTypeNames.FLOAT),				
                bigquery.SchemaField(config["decile_column_name"], bigquery.enums.SqlTypeNames.INTEGER),	
                bigquery.SchemaField(config["decile_consented_column_name"], bigquery.enums.SqlTypeNames.INTEGER),
                bigquery.SchemaField(config["model_version_column_name"], bigquery.enums.SqlTypeNames.STRING),				
                bigquery.SchemaField(config["kpis_run_date_column_name"], bigquery.enums.SqlTypeNames.DATETIME),				
                bigquery.SchemaField(config["run_date_column_name"], bigquery.enums.SqlTypeNames.DATETIME),				
                bigquery.SchemaField(config["prediction_reference_date_column_name"], bigquery.enums.SqlTypeNames.DATE)
            ]
    big_query_client = BigQueryClient(job_args["project_id"], logger)
    big_query_client.export(
        data=result_df,
        export_project_id=job_args["export_project_id"],
        export_dataset_id=job_args["export_dataset_id"],
        export_table_id=job_args["export_table_id"],
        export_table_schema=output_schema
    )
    logger.info("Final dataset written on BQ")


if __name__ == "__main__":
    job_args = parse_arguments()
    logger = create_logger('{}'.format(job_args["model_name"]),
                                                             job_args["project_id"])

    root_path = Path(__file__).parent
    config_path = root_path / 'config' / 'prediction'

    ######### Load config file #########
    config = read_yaml(config_path / "config_{}.yaml".format(job_args["model_name"]))

    # Init Folders
    output_folder, metadata_folder, dataset_folder, gcs_metadata_path = init_folders(job_args, config, root_path)

    # download set of columns used for training and preprocess config
    download_blob_file(job_args["project_id"], job_args["gcs_bucket"], gcs_metadata_path+'/training_config.yaml', metadata_folder + "/training_config.yaml")
    training_config = read_yaml(metadata_folder + "/training_config.yaml")
    preprocess_config = training_config["PREPROCESS"]

    # Preprocessing
    data, consent_col, date_kpis_run = data_preprocessing(job_args,
                                                            config,
                                                            training_config,
                                                            preprocess_config,
                                                            metadata_folder, 
                                                            dataset_folder, 
                                                            gcs_metadata_path,
                                                            logger)

    # Prediction
    prediction(data,
                consent_col,
                date_kpis_run,
                config, 
                training_config,
                metadata_folder,
                logger,
                job_args)
    

    
    
    
    
   

