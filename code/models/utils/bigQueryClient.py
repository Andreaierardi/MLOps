import logging

from google.cloud import bigquery

from google.api_core.exceptions import NotFound, BadRequest
import pandas as pd

class BigQueryClient():

    def __init__(self, project_id, log_format, timeout=120): 
        
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        logging.basicConfig(level=logging.DEBUG, format=log_format)
        
        self._dataset_ref = None
        self._table_ref= None
        
        self.logger = logging.getLogger(BigQueryClient.__name__)
        
        self.timeout = timeout
        
        self.logger.info("Big Query Client created using default project: {}".format(self.client.project))
        
    @property
    def dataset_ref(self):
        return self._dataset_ref

    @dataset_ref.setter
    def dataset_ref(self, arr):
        self._dataset_ref = self.client.dataset(arr[0], project = arr[1]) # set client reference to a particular dataset_ref
        
    @property
    def table_ref(self):
        return self._table_ref


    @table_ref.setter
    def table_ref(self, table_id):
        """
        
        table_id is string of following format "project.dataset.table"
        
        """
        self._table_ref = self.client.get_table(table_id) # set client reference to a particular table
    

    def run_query(self, query, query_parameters = [], return_df = False):
        
        
        """
        execute a sql query as python string
        """
        
        try:
            job_config = bigquery.QueryJobConfig(query_parameters = query_parameters)

            #self.logger.info("Running query")

            dataframe = self.client.query(query, retry = bigquery.DEFAULT_RETRY.with_deadline(self.timeout), job_config = job_config).result()

            self.logger.info(return_df)

            if return_df:
                return dataframe.to_dataframe()
            else:
                return
        
        except Exception as e:
            self.logger.error("BIG QUERY CLIENT ERROR LOADING: {}".format(e))
            raise Exception("BIG QUERY CLIENT ERROR LOADING: {}".format(e))
    
    def upload_datframe(self, df, table_name, project_id, if_exists="replace"):
        
        """
        upload a dataframe to Big Query

        """
        
        try:
            # Load data to BQ
            df.to_gbq(table_name, project_id= project_id, if_exists= if_exists)
            self.logger.info("TABLE UPLOADED {}".format(self.client.project))

        except BadRequest as e:
            self.logger.error("BIG QUERY CLIENT ERROR WRITING: {}".format(e))
            raise Exception("BIG QUERY CLIENT ERROR WRITING: {}".format(e))

    def export(self,
               data: pd.DataFrame,
               export_project_id: str,
               export_dataset_id: str,
               export_table_id: str,
               export_table_schema: list,
               partitioning_field: str=None):

        destination = export_project_id + '.' + export_dataset_id + '.' + export_table_id

        job_config = bigquery.LoadJobConfig()
        job_config.source_format = 'CSV'
        job_config.schema = export_table_schema
        if partitioning_field:
            job_config.write_disposition = 'WRITE_APPEND'
            job_config.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY,
                                                                     field=partitioning_field)
        else:
            job_config.write_disposition = 'WRITE_TRUNCATE'  
            job_config.create_disposition = 'CREATE_IF_NEEDED' 

        job = self.client.load_table_from_dataframe(
            dataframe=data,
            destination=destination,
            job_config=job_config
        )
        # Wait for the load job to complete.
        job.result()
