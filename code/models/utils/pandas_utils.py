import numpy as np
import pandas as pd
import gc 
from tqdm import tqdm

str_to_type_mapping = {
    "str" : str,
    "int" : int,
    "float" : float,
    "bool" : bool
}


def process_config_converters_dict(converters : dict) -> dict:
    """
    maps the values of converters to the python primitive class types
    e.g. {'column_name' : 'int'} --> {'column_name' : <class 'int'>}
    """
    output_dict = dict()
    if converters == None:
        return output_dict
    for converter in converters:
        try:
            output_dict[converter] = str_to_type_mapping[converters[converter]]
        except(KeyError):
            raise TypeError("Invalid converter type. Types accepted: str, int, float, bool. Received {}".format(converters[converter]))
    return output_dict

def memory_reduction(df:pd.DataFrame,
                     int_cast=True, 
                     obj_to_category=False, 
                     subset=None):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    gc.collect()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    cols = subset if subset is not None else df.columns.tolist()

    for col in tqdm(cols):
        col_type = df[col].dtype

        try:
            if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
                c_min = df[col].min()
                c_max = df[col].max()

                # test if column can be converted to an integer
                treat_as_int = str(col_type)[:3] == 'int'
                if int_cast and not treat_as_int:
                    treat_as_int = df[col].dtype == int

                if treat_as_int:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                    elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            elif 'datetime' not in col_type.name and obj_to_category:
                df[col] = df[col].astype('category')
        except:
            print(f'Feature "{col}" not reducible')

    gc.collect()
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.3f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df