import asyncio
from aiofile import AIOFile


async def main():
    async with AIOFile("/tmp/hello.txt", 'w+') as afp:
        await afp.write("Hello ")
        await afp.write("world", offset=7)
        await afp.fsync()

        print(await afp.read())


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
