import multiprocessing as mp
import multiprocessing.pool
import geopandas

files_to_read = ["/home/andy/workspace/nhm/nhm_hru_data/nhru_01.shp", "/home/andy/workspace/nhm/nhm_hru_data/nhru_02.shp", "/home/andy/workspace/nhm/nhm_hru_data/nhru_03.shp"]

# guessing a max of 4 threads would be reasonable since much of read_file
# will likely be done in a C extension without the GIL
pool=mp.pool.ThreadPool(min(mp.cpu_count, len(files_to_read), 4))
frames = pool.map(geopandas.read_file, files_to_read, chunksize=1)
pool.close()
