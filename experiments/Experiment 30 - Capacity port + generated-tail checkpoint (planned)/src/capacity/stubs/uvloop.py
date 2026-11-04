import asyncio
EventLoopPolicy = asyncio.DefaultEventLoopPolicy
def install():
    pass
def new_event_loop():
    return asyncio.new_event_loop()
def run(main, **kw):
    return asyncio.run(main)
