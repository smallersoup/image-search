import uuid
import logging
from towhee.operator import PyOperator, SharedType
from pymilvus import connections, Collection


logger = logging.getLogger()


class MilvusClient(PyOperator):
    """
    Milvus ANN index class.
    """

    def __init__(
        self, host: str = 'localhost', port: int = 19530, collection_name: str = None,  uri: str = None, user: str = None, password: str = None, token: str = None
    ):
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

    def __call__(self, *data):
        """
        Insert one row to Milvus.

        Args:
        data (`list`):
            The data to insert into milvus.

        Returns:
            A MutationResult object contains `insert_count` represents how many and a `primary_keys` of primary keys.

        """
        row = []
        for item in data:
            if isinstance(item, list):
                row.extend([[i] for i in item])
            else:
                row.append([item])
        mr = self._collection.insert(row)
        if mr.insert_count != len(row[0]):
            raise RuntimeError("Insert to milvus failed")
        return mr

    @property
    def shared_type(self):
        return SharedType.NotShareable

    # def __del__(self):
    #     if connections.has_connection(self._connect_name):
    #         try:
    #             connections.disconnect(self._connect_name)
    #         except:
    #             pass
