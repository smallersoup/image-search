import csv
from glob import glob
from pathlib import Path
import gradio as gr
from towhee import pipe, ops, DataCollection

# Towhee parameters
from reverse_image_search.tcvdb_client import TcvdbClient

MODEL = 'resnet50'
DEVICE = None  # if None, use default device (cuda is enabled if available)

# tcvdb parameters
HOST = 'lb-xxx.clb.ap-beijing.tencentclb.com'
PORT = '10000'
DB_NAME = 'image-search'
COLLECTION_NAME = 'reverse_image_search'

PASSWORD = 'xxxxx'
USERNAME = 'root'

# path to csv (column_1 indicates image path) OR a pattern of image paths
INSERT_SRC = 'reverse_image_search.csv'

# Load image path
def load_image(x):
    if x.endswith('csv'):
        with open(x) as f:
            reader = csv.reader(f)
            next(reader)
            for item in reader:
                yield item[1]
    else:
        for item in glob(x):
            yield item


def search_and_show_images(file_path):
    # 使用 `file_path` 进行搜索，返回结果的路径
    results = p_search(file_path)  # 假设 `p_search` 是你的搜索函数
    # 从 'DataQueue' 对象中获取数据
    data = results.get()

    # 获取 'pred' 字段的值，类如['/root/image-search/reverse_image_search/train/cuirass/n03146219_11082.JPEG',
    # '/root/image-search/reverse_image_search/train/loudspeaker/n03691459_40992.JPEG',
    # '/root/image-search/reverse_image_search/train/comic_book/n06596364_11921.JPEG',
    # '/root/image-search/reverse_image_search/train/ski_mask/n04229816_6821.JPEG',
    # '/root/image-search/reverse_image_search/train/traffic_light/n06874185_15185.JPEG',
    # '/root/image-search/reverse_image_search/train/tiger_cat/n02123159_6503.JPEG',
    # '/root/image-search/reverse_image_search/train/dishwasher/n03207941_15169.JPEG',
    # '/root/image-search/reverse_image_search/train/steam_locomotive/n04310018_10624.JPEG',
    # '/root/image-search/reverse_image_search/train/minibus/n03769881_619.JPEG',
    # '/root/image-search/reverse_image_search/train/apiary/n02727426_948.JPEG']
    pred = data[1]
    return pred


if __name__ == '__main__':
    # test_vdb = Demo_TCVDB('vdb http url or ip and post', key='key get from web console', username='vdb username')
    test_vdb = TcvdbClient(host=HOST, port=PORT, key='xxxxx', username='root',
                           collectionName=COLLECTION_NAME, dbName=DB_NAME)
    # test_vdb.clear()  # 测试前清理环境
    # test_vdb.create_db_and_collection()

    # Embedding pipeline
    p_embed = (
        pipe.input('src')
            .flat_map('src', 'img_path', load_image)
            .map('img_path', 'img', ops.image_decode())
            .map('img', 'vec', ops.image_embedding.timm(model_name=MODEL, device=DEVICE))
    )

    # Display embedding result, no need for implementation
    p_display = p_embed.output('img_path', 'img', 'vec')
    DataCollection(p_display('./test/goldfish/*.JPEG')).show()

    # Insert pipeline
    p_insert = (
        p_embed.map(('img_path', 'vec'), 'mr', ops.local.tcvdb_client(
            host=HOST, port=PORT, key=PASSWORD, username=USERNAME,
            collectionName=COLLECTION_NAME, dbName=DB_NAME
        ))
            .output('mr')
    )

    # Insert data
    p_insert(INSERT_SRC)

    # Search pipeline
    p_search_pre = (
        p_embed.map('vec', ('search_res'), ops.local.search_tcvdb_client(
            host=HOST, port=PORT, key=PASSWORD, username=USERNAME,
            collectionName=COLLECTION_NAME, dbName=DB_NAME))
            .map('search_res', 'pred', lambda x: [str(Path(y[0]).resolve()) for y in x])
        # .output('img_path', 'pred')
    )
    p_search = p_search_pre.output('img_path', 'pred')

    # Search for example query image(s)
    # dc = p_search('test/goldfish/*.JPEG')
    # DataCollection(dc).show()

    iface = gr.Interface(
        fn=search_and_show_images,
        # inputs=gr.inputs.File(type="file"),
        inputs=gr.inputs.Textbox(default='test/goldfish/*.JPEG'),
        outputs=gr.Gallery(label="最终的结果图片").style(height='auto', columns=4),
        title='Tencent vector db 案例: 以图搜图',
    )
    iface.launch()
    #
    # for row in dc.get():
    #     print(row)
    # test_vdb.upsert_data_test()
    # test_vdb.query_data()
    # test_vdb.update_and_delete()
    # test_vdb.delete_and_drop()
