from .milvus_client import MilvusClient

def milvus_client(*args, **kwargs):
    return MilvusClient(*args, **kwargs)
