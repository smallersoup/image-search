import logging
import uuid

import numpy as np
from towhee.operator import PyOperator, SharedType
import json
import tcvectordb
from tcvectordb.model.collection import Embedding
from tcvectordb.model.enum import FieldType, IndexType, MetricType, EmbeddingModel, ReadConsistency
from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams, IVFFLATParams
from tcvectordb.model.document import Document, Filter, SearchParams

logger = logging.getLogger()


def print_object(obj):
    for elem in obj:
        if hasattr(elem, '__dict__'):
            print(json.dumps(vars(elem), indent=2))
        else:
            print(json.dumps(elem, indent=2))


class TcvdbClient(PyOperator):

    def __init__(self, host: str, port: str, username: str, key: str, dbName: str, collectionName: str,
                 timeout: int = 20):
        """
        初始化客户端
        """
        # 创建客户端时可以指定 read_consistency，后续调用 sdk 接口的 read_consistency 将延用该值
        self.collectionName = collectionName
        self.db_name = dbName
        self._client = tcvectordb.VectorDBClient(url="http://" + host + ":" + port, username=username, key=key,
                                                 read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY, timeout=timeout)

    def clear(self):
        db = self._client.database(self.db_name)
        db.drop_database(self.db_name)

    def delete_and_drop(self):
        db = self._client.database(self.db_name)

        # 删除collection，删除collection的同时，其中的数据也将被全部删除
        db.drop_collection(self.collectionName)

        # 删除db，db下的所有collection都将被删除
        db.drop_database(self.db_name)

    def create_db_and_collection(self):
        database = self.db_name
        coll_embedding_name = self.collectionName
        coll_alias = self.collectionName + "-alias"

        # 创建DB
        db = self._client.create_database(database)

        database_list = self._client.list_databases()
        for db_item in database_list:
            print(db_item.database_name)

        # 新建 Collection
        # 第一步，设计索引（不是设计表格的结构）
        # 1. 【重要的事】向量对应的文本字段不要建立索引，会浪费较大的内存，并且没有任何作用。
        # 2. 【必须的索引】：主键 id、向量字段 vector 这两个字段目前是固定且必须的，参考下面的例子；
        # 3. 【其他索引】：检索时需作为条件查询的字段，比如要按书籍的作者进行过滤，这个时候author字段就需要建立索引，
        #     否则无法在查询的时候对 author 字段进行过滤，不需要过滤的字段无需加索引，会浪费内存；
        # 4.  向量数据库支持动态 Schema，写入数据时可以写入任何字段，无需提前定义，类似 MongoDB.
        # 5.  例子中创建一个书籍片段的索引，例如书籍片段的信息包括 {id, vector, segment, bookName, page},
        #     id 为主键需要全局唯一，segment 为文本片段, vector 为 segment 的向量，vector 字段需要建立向量索引，假如我们在查询的时候要查询指定书籍
        #     名称的内容，这个时候需要对bookName建立索引，其他字段没有条件查询的需要，无需建立索引。
        # 6.  创建带 Embedding 的 collection 需要保证设置的 vector 索引的维度和 Embedding 所用模型生成向量维度一致，模型及维度关系：
        #     -----------------------------------------------------
        #             bge-base-zh                 ｜ 768
        #             m3e-base                    ｜ 768
        #             text2vec-large-chinese      ｜ 1024
        #             e5-large-v2                 ｜ 1024
        #             multilingual-e5-base        ｜ 768
        #     -----------------------------------------------------
        index = Index()
        index.add(VectorIndex('vector', 2048, IndexType.HNSW, MetricType.COSINE, HNSWParams(m=16, efconstruction=200)))
        index.add(FilterIndex('id', FieldType.String, IndexType.PRIMARY_KEY))
        # index.add(FilterIndex('bookName', FieldType.String, IndexType.FILTER))
        index.add(FilterIndex('path', FieldType.String, IndexType.FILTER))

        # ebd = Embedding(vector_field='vector', field='text', model=EmbeddingModel.BGE_BASE_ZH)

        # 第二步：创建 Collection
        # 创建支持 Embedding 的 Collection
        db.create_collection(
            name=coll_embedding_name,
            shard=3,
            replicas=0,
            description='image embedding collection',
            index=index,
            # embedding=ebd,
            embedding=None,
            timeout=20
        )

        print(f'A new collection created --- : {self.collectionName}')

        # 列出所有 Collection
        coll_list = db.list_collections()
        print_object(coll_list)

        # 设置 Collection 的 alias
        db.set_alias(coll_embedding_name, coll_alias)

        # 查看 Collection 信息
        coll_res = db.describe_collection(coll_embedding_name)
        print(vars(coll_res))

        # 删除 Collection 的 alias
        db.delete_alias(coll_alias)

    def upsert_data(self, document_list):
        # 获取 Collection 对象
        db = self._client.database(self.db_name)
        coll = db.collection(self.collectionName)
        # upsert 写入数据，可能会有一定延迟
        # 1. 支持动态 Schema，除了 id、vector 字段必须写入，可以写入其他任意字段；
        # 2. upsert 会执行覆盖写，若文档id已存在，则新数据会直接覆盖原有数据(删除原有数据，再插入新数据)
        coll.upsert(documents=document_list)

    def upsert_data_test(self):
        # 获取 Collection 对象
        db = self._client.database(self.db_name)
        coll = db.collection(self.collectionName)

        # upsert 写入数据，可能会有一定延迟
        # 1. 支持动态 Schema，除了 id、vector 字段必须写入，可以写入其他任意字段；
        # 2. upsert 会执行覆盖写，若文档 id 已存在，则新数据会直接覆盖原有数据(删除原有数据，再插入新数据)

        document_list = [
            Document(id='0001',
                     text='富贵功名，前缘分定，为人切莫欺心。',
                     bookName='西游记',
                     author='吴承恩',
                     page=21,
                     segment='富贵功名，前缘分定，为人切莫欺心。'),

        ]
        coll.upsert(documents=document_list)

    def __call__(self, *data):
        """
        Insert one row to Tcvdb.

        Args:
        data (`list`):
            The data to insert into Tcvdb.

        Returns:
            A MutationResult object contains `insert_count` represents how many and a `primary_keys` of primary keys.

        """
        path = ""
        vector = []
        for item in data:
            if isinstance(item, np.ndarray):
                vector = list(map(float, item))  # Convert ndarray to list and float32 to float
            else:
                path = item

        # Generate a random UUID and convert to string
        document_list = [
            Document(
                id=str(uuid.uuid4()),
                path=path,
                vector=vector),
        ]

        self.upsert_data(document_list)
        return

    @property
    def shared_type(self):
        return SharedType.NotShareable

    # def __del__(self):
    #     if connections.has_connection(self._connect_name):
    #         try:
    #             connections.disconnect(self._connect_name)
    #         except:
    #             pass

#
# class TcvdbClient(PyOperator):
#     """
#     Tcvdb ANN index class.
#     """
#
#     def __init__(
#             self, host: str = '10.0.0.1', port: int = 80, collection_name: str = None, uri: str = None,
#             user: str = None, password: str = None
#     ):
#         self._host = host
#         self._port = port
#         self._uri = uri
#         self._collection_name = collection_name
#         self._connect_name = uuid.uuid4().hex
#         if user and password:
#             # 创建客户端时可以指定 read_consistency，后续调用 sdk 接口的 read_consistency 将延用该值
#             self._client = tcvectordb.VectorDBClient(url=host, username=user, key=password,
#                                                      read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
#                                                      timeout=20)
#         else:
#             connections.connect(alias=self._connect_name, host=self._host, port=self._port)
#
#         self._collection = Collection(self._collection_name, using=self._connect_name)
#
#     def __call__(self, *data):
#         """
#         Insert one row to Tcvdb.
#
#         Args:
#         data (`list`):
#             The data to insert into tcvdb.
#
#         Returns:
#             A MutationResult object contains `insert_count` represents how many and a `primary_keys` of primary keys.
#
#         """
#         row = []
#         for item in data:
#             if isinstance(item, list):
#                 row.extend([[i] for i in item])
#             else:
#                 row.append([item])
#         mr = self._collection.insert(row)
#         if mr.insert_count != len(row[0]):
#             raise RuntimeError("Insert to tcvdb failed")
#         return mr
#
#     @property
#     def shared_type(self):
#         return SharedType.NotShareable
#
#     # def __del__(self):
#     #     if connections.has_connection(self._connect_name):
#     #         try:
#     #             connections.disconnect(self._connect_name)
#     #         except:
#     #             pass
