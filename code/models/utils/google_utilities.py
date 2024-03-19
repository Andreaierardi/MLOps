from google.cloud import storage
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, BadRequest

import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler

from datetime import datetime
import os
from pathlib import Path
import socket
from time import time

def upload_output(project, source_file_name, model_name):
    """Uploads a file to the bucket."""
    start = time()

    seed = 42
    today = datetime.today().strftime('%Y-%m-%d')

    bucket_name = project + "-automation"
    now = datetime.now()

    timestamp = datetime.timestamp(now)
    destination_blob_name = "output/"+model_name+"/"+today+"/"+str(timestamp)+"-"+os.path.basename(source_file_name)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def upload_file_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def upload_folder_to_gcs(bucket_name, source_folder_name, destination_blob_name):
    """ 
    Upload a folder to a bucket in GCS. 
    WARNING: it works ONLY if the folder contains files, not nested folders
    """
    pathlist = Path(source_folder_name).glob('*')
    for single_path in pathlist:
        upload_file_to_gcs( bucket_name,
                            single_path,
                            destination_blob_name + '/' + single_path.parent.name + '/' + single_path.name)

def get_export_bucket_name(project_id, environment):
    export_bucket = project_id + "-export"
    if environment == "test":
        export_bucket = export_bucket + "-test"
    return export_bucket

def get_export_dataset_name(project_id, environment, dataset_prefix):
    export_dataset = dataset_prefix
    if environment == "test":
        export_dataset = export_dataset + "_test"
    return export_dataset

def get_absolute_csv_path(project_id, flow_name, date, date_ext):
    export_path = flow_name + "/date=" + str(date) + "/" + flow_name + "_" + str(date_ext)
    return export_path

def getExportDataLocation(project_id, environment, flow_name):
    flow_path = "gs://" + get_export_bucket_name(project_id, environment) + "/" + flow_name
    return flow_path

def upload_csv_output(project_id, environment, source_file_name, flow_name):
    """Uploads a file to the bucket."""
    start = time()
    today = datetime.today().strftime('%Y-%m-%d')
    now = datetime.now()
    timestamp = datetime.timestamp(now)

    bucket_name = get_export_bucket_name(project_id, environment)
    destination_blob_name = get_absolute_csv_path(project_id, flow_name, today, timestamp)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def download_blob_file(
        project_id,
        bucket_name,
        source_blob_name,
        local_blob_name
):
    """General function to downloads a blob from the bucket."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(local_blob_name)

def download_input(project, input_path):
    """Downloads a blob from the bucket."""
    bucket_name = project + "-automation"
    source_blob_name = "input/"+input_path
    destination_file_name = "input.csv"

    download_blob_file(project, bucket_name, source_blob_name, destination_file_name)

def download_config(
        project_id,
        flow_name,
        environment,
        filename
):
    """Downloads a blob from the bucket."""
    bucket_name = project_id + "-automation"
    source_prefix = flow_name + "/" + environment + "/config/"
    local_prefix = "config/"
    source_blob_name = source_prefix + filename
    local_blob_name = local_prefix + filename

    download_blob_file(project_id, bucket_name, source_blob_name, local_blob_name)
    return(local_blob_name)

def download_model(
        project_id,
        flow_name,
        environment,
        filename
):
    """Downloads a blob from the bucket."""
    bucket_name = project_id + "-automation"
    source_prefix = flow_name + "/" + environment + "/model/"
    local_prefix = "model/"
    source_blob_name = source_prefix + filename
    local_blob_name = local_prefix + filename

    download_blob_file(project_id, bucket_name, source_blob_name, local_blob_name)
    return(local_blob_name)

def download_data_validation(
        project_id,
        flow_name,
        environment,
        filename
):
    """Downloads a blob from the bucket."""
    bucket_name = project_id + "-automation"
    source_prefix = flow_name + "/" + environment + "/data_validation/"
    local_prefix = "data_validation/"
    source_blob_name = source_prefix + filename
    local_blob_name = local_prefix + filename

    download_blob_file(project_id, bucket_name, source_blob_name, local_blob_name)
    return(local_blob_name)

def get_bq_retrier():
    new_timeout = 120
    return bigquery.DEFAULT_RETRY.with_deadline(new_timeout)

def table_exists(client, dataset_id, table_id):
    dataset_ref = client.dataset(dataset_id)
    try:
        table = client.get_table(bigquery.Table(dataset_ref.table(table_id)))
        if (table.created != None):
            return True
        else:
            return False

    except NotFound:
        print("The table {}.{} doesnt exists".format(dataset_id, table_id))
        return False

def create_external_partitioned_table(project_id, environment,
                                      dataset_prefix, flow_name,
                                      cols_name_schema):

    dataset_id = get_export_dataset_name(project_id, environment, dataset_prefix)
    table_id = flow_name

    client = bigquery.Client(project=project_id)
    if(not table_exists(client, dataset_id, table_id)):
        dataset_ref = client.dataset(dataset_id)
        table = bigquery.Table(dataset_ref.table(table_id))
        prefix_path = getExportDataLocation(project_id, environment, flow_name) + "/"
        dataPath = prefix_path + "*"

        def schema_col_def(name):
            return bigquery.SchemaField(name=name, field_type='STRING')

        external_config = bigquery.ExternalConfig("CSV")
        # external_config.autodetect = True
        external_config.schema = map(schema_col_def, cols_name_schema)
        external_config.options.field_delimiter = ","
        external_config.options.skip_leading_rows = 1

        external_config.source_uris = dataPath

        hive_partitioning = bigquery.external_config.HivePartitioningOptions()
        hive_partitioning.mode = "AUTO"
        hive_partitioning.source_uri_prefix = prefix_path

        external_config.hive_partitioning = hive_partitioning
        table.external_data_configuration = external_config

        print("creating")
        client.create_table(table, retry=get_bq_retrier())
        return True
    return False

def create_logger(logger_name, project_id):
    # function to create a logger for required process
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Cloud handler (Stackdriver)
    client = google.cloud.logging.Client(project=project_id)
    h1 = CloudLoggingHandler(client)
    logger.addHandler(h1)
    # Console handler
    h2 = logging.StreamHandler()
    h2.setLevel(logging.INFO)
    h2.setFormatter(logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s'))
    logger.addHandler(h2)
    return logger