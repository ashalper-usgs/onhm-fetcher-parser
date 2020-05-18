from pathlib import Path
import asyncio
import geopandas as gpd

async def read_file(name):
    df = gpd.read_file(name)
    print(df.head())

async def main():
    paths = Path('/home/andy/workspace/nhm/nhm_hru_data').glob('*.shp')
    tasks = []
    for path in paths:
        print(str(path))
        task = asyncio.ensure_future(read_file(str(path)))
        tasks.append(task)
    await asyncio.gather(*tasks)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
