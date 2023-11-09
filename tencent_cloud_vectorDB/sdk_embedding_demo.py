# example.py demonstrates the basic operations of tcvectordb, a Python SDK of tencent cloud vectordb.
# 1. connect to vectordb and create database and collection
# 2. upsert data
# 3. query and search data
# 4. drop collection and database
import json
import time

import tcvectordb
from tcvectordb.model.collection import Embedding
from tcvectordb.model.document import Document, Filter, SearchParams
from tcvectordb.model.enum import FieldType, IndexType, MetricType, EmbeddingModel, ReadConsistency
from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams, IVFFLATParams

# disable/enable http request log print
tcvectordb.debug.DebugEnable = False


def print_object(obj):
    for elem in obj:
        if hasattr(elem, '__dict__'):
            print(json.dumps(vars(elem), indent=2))
        else:
            print(json.dumps(elem, indent=2))


class TestVDB:

    def __init__(self, url: str, username: str, key: str, timeout: int = 30):
        """
        初始化客户端
        """
        # 创建客户端时可以指定 read_consistency，后续调用 sdk 接口的 read_consistency 将延用该值
        self._client = tcvectordb.VectorDBClient(url=url, username=username, key=key,
                                                 read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY, timeout=timeout)

    def clear(self):
        db = self._client.database('book')
        db.drop_database('book')

    def delete_and_drop(self):
        db = self._client.database('book')

        # 删除collection，删除collection的同时，其中的数据也将被全部删除
        db.drop_collection('book_segments')

        # 删除db，db下的所有collection都将被删除
        db.drop_database('book')

    def create_db_and_collection(self):
        database = 'book'
        coll_embedding_name = 'book_segments'
        coll_alias = 'book_segments_alias'

        # 创建DB--'book'
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
        index.add(VectorIndex('vector', 768, IndexType.HNSW, MetricType.COSINE, HNSWParams(m=16, efconstruction=200)))
        index.add(FilterIndex('id', FieldType.String, IndexType.PRIMARY_KEY))
        index.add(FilterIndex('bookName', FieldType.String, IndexType.FILTER))
        index.add(FilterIndex('author', FieldType.String, IndexType.FILTER))

        ebd = Embedding(vector_field='vector', field='text', model=EmbeddingModel.BGE_BASE_ZH)

        # 第二步：创建 Collection
        # 创建支持 Embedding 的 Collection
        db.create_collection(
            name=coll_embedding_name,
            shard=3,
            replicas=0,
            description='test embedding collection',
            index=index,
            embedding=ebd,
            timeout=20
        )

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

    def upsert_data(self):
        # 获取 Collection 对象
        db = self._client.database('book')
        coll = db.collection('book_segments')

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
            Document(id='0002',
                     text='正大光明，忠良善果弥深。些些狂妄天加谴，眼前不遇待时临。',
                     bookName='西游记',
                     author='吴承恩',
                     page=22,
                     segment='正大光明，忠良善果弥深。些些狂妄天加谴，眼前不遇待时临。'),
            Document(id='0003',
                     text='细作探知这个消息，飞报吕布。',
                     bookName='三国演义',
                     author='罗贯中',
                     page=23,
                     segment='细作探知这个消息，飞报吕布。'),
            Document(id='0004',
                     text='布大惊，与陈宫商议。宫曰：“闻刘玄德新领徐州，可往投之。”布从其言，竟投徐州来。有人报知玄德。',
                     bookName='三国演义',
                     author='罗贯中',
                     page=24,
                     segment='布大惊，与陈宫商议。宫曰：“闻刘玄德新领徐州，可往投之。”布从其言，竟投徐州来。有人报知玄德。'),
            Document(id='0005',
                     text='玄德曰：“布乃当今英勇之士，可出迎之。”糜竺曰：“吕布乃虎狼之徒，不可收留；收则伤人矣。',
                     bookName='三国演义',
                     author='罗贯中',
                     page=25,
                     segment='玄德曰：“布乃当今英勇之士，可出迎之。”糜竺曰：“吕布乃虎狼之徒，不可收留；收则伤人矣。'),

        ]
        coll.upsert(documents=document_list)
        time.sleep(1)

    def query_data(self):
        # 获取 Collection 对象
        db = self._client.database('book')
        coll = db.collection('book_segments')

        # 查询
        # 1. query 用于查询数据
        # 2. 可以通过传入主键 id 列表或 filter 实现过滤数据的目的
        # 3. 如果没有主键 id 列表和 filter 则必须传入 limit 和 offset，类似 scan 的数据扫描功能
        # 4. 如果仅需要部分 field 的数据，可以指定 output_fields 用于指定返回数据包含哪些 field，不指定默认全部返回

        document_ids = ["0001", "0002", "0003", "0004", "0005"]
        filter_param = Filter('bookName="三国演义"')
        output_fields_param = ["id", "bookName"]
        res = coll.query(document_ids=document_ids, retrieve_vector=False, limit=2, offset=1,
                         filter=filter_param, output_fields=output_fields_param)
        print_object(res)

        # searchById
        # 1. searchById 提供按 id 搜索的能力
        # 1. search 提供按照 vector 搜索的能力
        # 2. 支持通过 filter 过滤数据
        # 3. 如果仅需要部分 field 的数据，可以指定 output_fields 用于指定返回数据包含哪些 field，不指定默认全部返回
        # 4. limit 用于限制每个单元搜索条件的条数，如 vector 传入三组向量，limit 为 3，则 limit 限制的是每组向量返回 top 3 的相似度向量

        # 根据主键 id 查找 Top K 个相似性结果，向量数据库会根据ID 查找对应的向量，再根据向量进行TOP K 相似性检索
        res = coll.searchById(
            document_ids=['0003'],
            params=SearchParams(ef=200),  # 若使用HNSW索引，则需要指定参数ef，ef越大，召回率越高，但也会影响检索速度
            retrieve_vector=False,  # 是否需要返回向量字段，False：不返回，True：返回
            limit=2,  # 指定 Top K 的 K 值
            filter=filter_param  # 过滤获取到结果
        )
        print_object(res)

        # search
        # 1. search 提供按照 vector 搜索的能力
        # 其他选项类似 search 接口

        # 批量相似性查询，根据指定的多个向量查找多个 Top K 个相似性结果
        query_all = coll.query(document_ids=document_ids, retrieve_vector=True, limit=2)
        query_document_vector = [x.get("vector") for x in query_all]
        res = coll.search(
            vectors=query_document_vector,  # 指定检索向量，最多指定20个
            params=SearchParams(ef=200),  # 若使用HNSW索引，则需要指定参数ef，ef越大，召回率越高，但也会影响检索速度
            retrieve_vector=False,  # 是否需要返回向量字段，False：不返回，True：返回
            limit=2,  # 指定 Top K 的 K 值
            filter=filter_param  # 对搜索结果进行过滤
        )
        # 输出相似性检索结果，检索结果为二维数组，每一位为一组返回结果，分别对应search时指定的多个向量
        print_object(res)

        # 通过 embedding 文本搜索
        # 1. searchByText 提供基于 embedding 文本的搜索能力，会先将 embedding 内容做 Embedding 然后进行按向量搜索
        # 其他选项类似 search 接口

        # searchByText 返回类型为 Dict，接口查询过程中 embedding 可能会出现截断，如发生截断将会返回响应 warn 信息，如需确认是否截断可以
        # 使用 "warning" 作为 key 从 Dict 结果中获取警告信息，查询结果可以通过 "documents" 作为 key 从 Dict 结果中获取
        embeddingItems = ['细作探知这个消息，飞报吕布。']
        search_by_text_res = coll.searchByText(embeddingItems=embeddingItems,
                                               params=SearchParams(ef=200))
        print_object(search_by_text_res.get('documents'))

    def update_and_delete(self):
        # 获取 Collection 对象
        db = self._client.database('book')
        coll = db.collection('book_segments')

        # update
        # 1. update 提供基于 [主键查询] 和 [Filter 过滤] 的部分字段更新或者非索引字段新增

        # filter 限制仅会更新 id = "0003"
        document_ids = ["0001", "0003"]
        filter_param = Filter('bookName="三国演义"')
        update_doc = Document(page=28)
        coll.update(data=update_doc, document_ids=document_ids, filter=filter_param)

        # delete
        # 1. delete 提供基于 [主键查询] 和 [Filter 过滤] 的数据删除能力
        # 2. 删除功能会受限于 collection 的索引类型，部分索引类型不支持删除操作

        # filter 限制只会删除 id="0001" 成功
        filter_param = Filter('bookName="西游记"')
        coll.delete(document_ids=document_ids, filter=filter_param)

        # rebuild_index
        # 索引重建，重建期间不支持写入
        coll.rebuild_index()

        # truncate_collection
        # 清空 Collection
        time.sleep(5)
        truncate_res = db.truncate_collection('book_segments')
        print_object(truncate_res)


if __name__ == '__main__':
    # test_vdb = Demo_TCVDB('vdb http url or ip and post', key='key get from web console', username='vdb username')
    test_vdb = TestVDB('http://lb-xxx.clb.ap-beijing.tencentclb.com:10000', key='xxxxx', username='root')
    test_vdb.clear()  # 测试前清理环境
    test_vdb.create_db_and_collection()
    test_vdb.upsert_data()
    test_vdb.query_data()
    # test_vdb.update_and_delete()
    # test_vdb.delete_and_drop()

