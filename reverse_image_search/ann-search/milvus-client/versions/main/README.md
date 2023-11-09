# ANN Search Operator: MilvusClient

*author: junjie.jiangjjj*

<br />

## Desription
Search embedding in [Milvus](https://milvus.io/), **please make sure you have inserted data to Milvus Collection**.

<br />



## Code Example

> Please make sure you have inserted data into Milvus, refer to [ann_insert.milvus_client](https://towhee.io/ann-insert/milvus-client) and [load the collection](https://milvus.io/docs/v2.1.x/load_collection.md) to memory.


```python
from towhee import pipe, ops, DataCollection

p = pipe.input('text')  \
    .map('text', 'vec', ops.sentence_embedding.transformers(model_name='all-MiniLM-L12-v2'))  \
    .flat_map('vec', 'rows', ops.ann_search.milvus_client(host='127.0.0.1', port='19530', collection_name='text_db2', **{'output_fields': ['text']}))  \
    .map('rows', ('id', 'score', 'text'), lambda x: (x[0], x[1], x[2]))  \
    .output('id', 'score', 'text')

DataCollection(p('cat')).show()

# result:

```

```python
from towhee import pipe, ops

# search additional info url:
from towhee import pipe, ops, DataCollection

p = pipe.input('text')  \
    .map('text', 'vec', ops.sentence_embedding.transformers(model_name='all-MiniLM-L12-v2'))  \
    .map('vec', 'rows', ops.ann_search.milvus_client(host='127.0.0.1', port='19530', collection_name='text_db2', **{'output_fields': ['text']}))  \
    .output('rows')

DataCollection(p('cat')).show()
```

<br />



## Factory Constructor

Create the operator via the following factory method:

***ann_search.milvus_client(host='127.0.0.1', port='19530', collection_name='textdb')***


<br />
