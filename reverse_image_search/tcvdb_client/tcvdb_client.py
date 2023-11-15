import logging
import uuid
import json
import tcvectordb
from tcvectordb.model.document import Document
from tcvectordb.model.enum import FieldType, IndexType, MetricType, ReadConsistency
from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams

logger = logging.getLogger()


def print_object(obj):
    for elem in obj:
        if hasattr(elem, '__dict__'):
            print(json.dumps(vars(elem), indent=2))
        else:
            print(json.dumps(elem, indent=2))


class TcvdbClient:

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

    # def upsert_data_test(self):
    #     # 获取 Collection 对象
    #     db = self._client.database(self.db_name)
    #     coll = db.collection(self.collectionName)
    #
    #     # upsert 写入数据，可能会有一定延迟
    #     # 1. 支持动态 Schema，除了 id、vector 字段必须写入，可以写入其他任意字段；
    #     # 2. upsert 会执行覆盖写，若文档 id 已存在，则新数据会直接覆盖原有数据(删除原有数据，再插入新数据)
    #
    #     document_list = [
    #         Document(id='0001',
    #                  text='富贵功名，前缘分定，为人切莫欺心。',
    #                  bookName='西游记',
    #                  author='吴承恩',
    #                  page=21,
    #                  segment='富贵功名，前缘分定，为人切莫欺心。'),
    #
    #     ]
    #     coll.upsert(documents=document_list)

    def upsert(self, path, item):
        """
        Insert one row to Tcvdb.

        Args:
        path (`str`):
            The path of the image to insert into Tcvdb.
        item (`np.ndarray`):
            The feature vector of the image.
        """
        # Convert ndarray to list and float32 to float
        vector = list(map(float, item.ravel()))
        # for item in data:
        #     if isinstance(item, np.ndarray):
        #         # Convert ndarray to list and float32 to float
        #         # vector = list(map(float, item))
        #         # Flatten the array and convert it to a list of floats
        #         vector = list(map(float, item.ravel()))
        #     else:
        #         path = item

        # Generate a random UUID and convert to string
        document_list = [
            Document(
                id=str(uuid.uuid4()),
                path=path,
                vector=vector),
        ]

        self.upsert_data(document_list)
        return

    def search(self, query: 'ndarray'):
        tcvdb_result = self.query_data(query)

        result = []
        # 过滤只显示指定的变量
        for hit in tcvdb_result[0]:
            row = []
            row.extend([hit["path"], hit["score"]])
            result.append(row)
        return result

    def query_data(self, query: []):
        # 获取 Collection 对象
        db = self._client.database(self.db_name)
        coll = db.collection(self.collectionName)
        vector = query
        # if isinstance(query, np.ndarray):
        #     vector = list(map(float, query))  # Convert ndarray to list and float32 to float
        vector = list(map(float, query.ravel()))
        # print(type(vector))
        # print(type(vector[0]))
        # print(vector)

        # search
        # 1. search 提供按照 vector 搜索的能力
        # 其他选项类似 search 接口

        # 批量相似性查询，根据指定的多个向量查找多个 Top K 个相似性结果
        # query_all = coll.query(document_ids=document_ids, retrieve_vector=True, limit=2)
        # query_document_vector = [x.get("vector") for x in query_all]
        res = coll.search(
            vectors=[vector],  # 指定检索向量，最多指定20个
            # params=SearchParams(ef=200),  # 若使用HNSW索引，则需要指定参数ef，ef越大，召回率越高，但也会影响检索速度
            retrieve_vector=False,  # 是否需要返回向量字段，False：不返回，True：返回
            limit=10,  # 指定 Top K 的 K 值
            # filter=filter_param  # 对搜索结果进行过滤
        )
        # 输出相似性检索结果，检索结果为二维数组，每一位为一组返回结果，分别对应search时指定的多个向量
        print_object(res)
        return res

        # 通过 embedding 文本搜索
        # 1. searchByText 提供基于 embedding 文本的搜索能力，会先将 embedding 内容做 Embedding 然后进行按向量搜索
        # 其他选项类似 search 接口

        # searchByText 返回类型为 Dict，接口查询过程中 embedding 可能会出现截断，如发生截断将会返回响应 warn 信息，如需确认是否截断可以
        # 使用 "warning" 作为 key 从 Dict 结果中获取警告信息，查询结果可以通过 "documents" 作为 key 从 Dict 结果中获取
        # embeddingItems = ['细作探知这个消息，飞报吕布。']
        # search_by_text_res = coll.searchByText(embeddingItems=embeddingItems,
        #                                        params=SearchParams(ef=200))
        # print_object(search_by_text_res.get('documents'))