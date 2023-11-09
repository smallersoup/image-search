from pymilvus import connections, Collection
from towhee.operator import PyOperator, SharedType
import uuid


class MilvusClient(PyOperator):
    """
    Search for embedding vectors in Milvus. Note that the Milvus collection has data before searching,

    Args:
        collection (`str`):
            The collection name.
        kwargs
            The kwargs with collection.search, refer to https://milvus.io/docs/v2.0.x/search.md#Prepare-search-parameters.
            And the `anns_field` defaults to the vector field name, `limit` defaults to 10, and `metric_type` in `param` defaults to 'L2'
            if there has no index(FLAT), and for default index `param`:
                IVF_FLAT: {"params": {"nprobe": 10}},
                IVF_SQ8: {"params": {"nprobe": 10}},
                IVF_PQ: {"params": {"nprobe": 10}},
                HNSW: {"params": {"ef": 10}},
                IVF_HNSW: {"params": {"nprobe": 10, "ef": 10}},
                RHNSW_FLAT: {"params": {"ef": 10}},
                RHNSW_SQ: {"params": {"ef": 10}},
                RHNSW_PQ: {"params": {"ef": 10}},
                ANNOY: {"params": {"search_k": 10}}.
    """

    def __init__(self, host: str = 'localhost', port: int = 19530, collection_name: str = None,
                 uri: str = None, user: str = None, password: str = None, token: str = None, **kwargs):
        """
        Get an existing collection.
        """
        self._host = host
        self._port = port
        self._uri = uri
        self._collection_name = collection_name
        self._connect_name = uuid.uuid4().hex
        if uri and token:
            connections.connect(alias=self._connect_name, uri=self._uri, token=token, secure=True)
        elif user and password:
            connections.connect(alias=self._connect_name, host=self._host, port=self._port,
                                user=user, password=password, secure=True)
        else:
            connections.connect(alias=self._connect_name, host=self._host, port=self._port)
        
    
        self._collection = Collection(self._collection_name, using=self._connect_name)      

        self.kwargs = kwargs
        if 'anns_field' not in self.kwargs:
            fields_schema = self._collection.schema.fields
            for schema in fields_schema:
                if schema.dtype in (101, 100):
                    self.kwargs['anns_field'] = schema.name

        if 'limit' not in self.kwargs:
            self.kwargs['limit'] = 10

        index_params = {
            'FLAT': {'params': {'nprobe': 10}},
            'IVF_FLAT': {'params': {'nprobe': 10}},
            'IVF_SQ8': {'params': {'nprobe': 10}},
            'IVF_PQ': {'params': {'nprobe': 10}},
            'HNSW': {'params': {'ef': 10}},
            'RHNSW_FLAT': {'params': {'ef': 10}},
            'RHNSW_SQ': {'params': {'ef': 10}},
            'RHNSW_PQ': {'params': {'ef': 10}},
            'IVF_HNSW': {'params': {'nprobe': 10, 'ef': 10}},
            'ANNOY': {'params': {'search_k': 10}},
            'AUTOINDEX': {}
        }

        if 'param' not in self.kwargs:
            if len(self._collection.indexes) != 0:
                index_type = self._collection.indexes[0].params['index_type']
                self.kwargs['param'] = index_params[index_type]
            else:
                self.kwargs['param'] = index_params['IVF_FLAT']
            if 'metric_type' in self.kwargs:
                self.kwargs['param']['metric_type'] = self.kwargs['metric_type']
            else:
                self.kwargs['param']['metric_type'] = 'L2'

    def __call__(self, query: 'ndarray'):
        self._collection.load()
        milvus_result = self._collection.search(
            data=[query],
            **self.kwargs
        )

        result = []
        for hit in milvus_result[0]:
            row = []
            row.extend([hit.id, hit.score])
            if 'output_fields' in self.kwargs:
                for k in self.kwargs['output_fields']:
                 row.append(hit.entity.get(k))
            result.append(row)
        return result

    @property
    def shared_type(self):
        return SharedType.NotShareable

    # def __del__(self):
    #     if connections.has_connection(self._connect_name):
    #         try:
    #             connections.disconnect(self._connect_name)
    #         except:
    #             pass
