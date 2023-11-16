import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import vertica_python
import pandas as pd
from typing import List, Tuple, Union, Dict
from itertools import product


def get_absolute_path(
        filename:str# = TRAIN_DATA_FILE
        , relative_path:str# = DATA_RELATIVE_PATH
        , base_dir:str# = os.path.expanduser(PROJECT_BASE_DIR)
        
):
    return os.path.join(base_dir, relative_path, filename)



def run_query(query: str, return_df: bool, query_args: dict) -> pd.DataFrame: 
    """
    Pull data from Chewy database.
    
    Input: 
        - query: a string of SQL query
        - return_df: a boolean value indicating the data format
        - query_args: other arguments used inside the query, for example, the dates.

    Example:
    chewy_df = run_query(
        query = 'select * FROM chewybi.inventory_snapshot limit '{sample_size}''
        , return_df = True
        , query_args = {'sample_size': 3}
    )

    print(chewy_df.head(3))    
            
    Note:
    source ~/.zshrc
    echo $VERTICA_USER
    echo $VERTICA_PASSWORD
    
    """
    
    # Adding your username + password + SQL script
    user = os.environ.get('VERTICA_USER')
    password = os.environ.get('VERTICA_PASSWORD')
    conn_info = {'host': 'bidb.chewy.local',
                 'port': 5433,
                 'user': user,
                 'password': password,
                 'database': 'bidb',
                 'read_timeout': 15000,
                 'connection_timeout': 10000
                 }

    connection = vertica_python.connect(**conn_info)
    cursor = connection.cursor()
    
    df = pd.DataFrame()
    if return_df == True:
        df = pd.read_sql(query.format(**query_args), connection, chunksize=None)
        connection.commit()
    else:
        cursor.execute(query.format(**query_args))
        connection.commit()
    connection.close()
    return df

