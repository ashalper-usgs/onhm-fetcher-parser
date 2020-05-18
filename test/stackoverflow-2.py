import asyncio
from pathlib import Path

async def run_check(shell_command):
    p = await asyncio.create_subprocess_shell(shell_command,
                    stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    fut = p.communicate()
    try:
        pcap_run = await asyncio.wait_for(fut, timeout=5)
    except asyncio.TimeoutError:
        p.kill()
        await p.communicate()

def get_shapefiles():
    filenames = Path('/home/andy/workspace/nhm/nhm_hru_data').glob('*.shp')
    for pcap_loc in filenames:
        for pcap_check in get_pcap_executables():
            tmp_coro = (run_check('{args}'
            .format(e=sys.executable, args=args)))
            if tmp_coro != False:
                shapefiles.append(tmp_coro)
     return shapefiles

async def main(self):
    ## Here p_shapefiles has over 500 files
    p_shapefiles = get_shapefiles()
    for f in asyncio.as_completed(p_shapefiles):
        res = await f

loop = asyncio.get_event_loop()
loop.run_until_complete(get_shapefiles())
loop.close()
