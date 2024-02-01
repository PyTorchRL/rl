import ray
from utils import SyncDataCollector

ray.init()
ray.remote(SyncDataCollector)
