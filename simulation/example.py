import psutil

from common.resource_profiler import profiled_transmission


def probe():
    cpu = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return cpu, memory


@profiled_transmission(body_arg='body', endpoint_arg='endpoint')
def test(body: dict, endpoint: str, reps=10 ** 6):
    for i in range(reps):
        var = 1
    str1 = 'python' * reps
    str2 = 'programmer' * reps
    str3 = str1 + str2
    del str2
    return str3


test(body={'foo': 'bar'}, endpoint='url')
