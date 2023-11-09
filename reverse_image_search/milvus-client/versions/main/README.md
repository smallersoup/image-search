# ANN Insert Operator: MilvusClient


<br />



## Desription

Insert data into Milvus collections. **Please make sure you have [created Milvus Collection](https://milvus.io/docs/v2.0.x/create_collection.md) before loading the data.**

<br />



## Code Example

### Example


```python
import towhee

from towhee import ops

p = (
        towhee.pipe.input('vec')
        .map('vec', (), ops.ann_insert.milvus_client(host='127.0.0.1', port='19530', collection_name='test_collection'))
        .output()
        )

p(vec)
```

### Load Collection

> Please load the Collection after inserted data.

```python
collection.load()
```

<br />



## Factory Constructor

Create the operator via the following factory method:

***ann_insert.milvus_client(host, port, collection_name, user= None, password=None, collection_schema=None, index_params=None)***

**Parameters:**

***host:*** *str*

The host for Milvus.

***port:*** *str*

The port for Milvus.

***collection_name:*** *str*

The collection name for Milvus.

***user:*** *str*

The user for Zilliz Cloud, defaults to None.

***password:*** *str*

The password for Zilliz Cloud, defaults to None.



<br />



## Interface

Insert Milvus data.

**Parameters:**

***data:*** *list*

The data to insert into milvus.

**Returns:** MutationResult

A MutationResult object contains `insert_count` represents how many and a `primary_keys` of primary keys.
