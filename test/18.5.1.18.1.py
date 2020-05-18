import asyncio

# See
# https://docs.python.org/3.6/library/asyncio-eventloop.html#hello-world-with-call-soon

def hello_world(loop):
    print('Hello World')
    loop.stop()

loop = asyncio.get_event_loop()

# Schedule a call to hello_world()
loop.call_soon(hello_world, loop)

# Blocking call interrupted by loop.stop()
loop.run_forever()
loop.close()
