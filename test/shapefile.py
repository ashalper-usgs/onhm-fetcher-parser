from multiprocessing import Pool, TimeoutError
from pathlib import Path
import os
import sys
import time
import geopandas as gpd

# See
# https://docs.python.org/3.6/library/multiprocessing.html#using-a-pool-of-workers

def gpd_read_file(filename):
    return gpd.read_file(filename)

if __name__ == '__main__':
    filenames = Path('/home/andy/workspace/nhm/nhm_hru_data').glob('*.shp')
    
    # start 4 worker processes
    with Pool(processes=4) as pool:
        for f in filenames:
            print(str(f))
            df = pool.map(gpd_read_file, str(f))
            df = gpd_read_file(str(f))
            print(df.head())

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
