import numpy as np
import json
import tcvectordb
from tcvectordb.model.enum import FieldType, IndexType, MetricType, EmbeddingModel, ReadConsistency


def print_object(obj):
    for elem in obj:
        if hasattr(elem, '__dict__'):
            print(json.dumps(vars(elem), indent=2))
        else:
            print(json.dumps(elem, indent=2))


class SearchTcvdbClient():
    def __init__(self, host: str, port: str, username: str, key: str, dbName: str, collectionName: str,
                 timeout: int = 20, **kwargs):
        """
        初始化客户端
        """
        # 创建客户端时可以指定 read_consistency，后续调用 sdk 接口的 read_consistency 将延用该值
        self.collectionName = collectionName
        self.db_name = dbName
        self._client = tcvectordb.VectorDBClient(url="http://" + host + ":" + port, username=username, key=key,
                                                 read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY, timeout=timeout)
        self.kwargs = kwargs

        if 'limit' not in self.kwargs:
            self.kwargs['limit'] = 10

    def __call__(self, query: 'ndarray'):
        tcvdb_result = self.query_data(
            query,
            **self.kwargs
        )

        result = []
        # 过滤只显示指定的变量
        for hit in tcvdb_result[0]:
            row = []
            row.extend([hit["path"], hit["score"]])
            result.append(row)
        return result

    def query_data(self, query: [], **kwargs):
        # 获取 Collection 对象
        db = self._client.database(self.db_name)
        coll = db.collection(self.collectionName)
        vector = query
        if isinstance(query, np.ndarray):
            vector = list(map(float, query))  # Convert ndarray to list and float32 to float

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
