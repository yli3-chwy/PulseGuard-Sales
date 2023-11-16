import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import pandas as pd
from itertools import product
import re
import pickle


from utils.constant import *
from utils.utils import get_absolute_path, run_query


# Load the data from database

input_script = get_absolute_path(
    filename="mega_sales.txt",
    relative_path='query',
    base_dir=PROJECT_BASE_DIR
)

input_df = run_query(
    input_script, True, query_args={'days': 356}
)

print(input_df.head(3))

