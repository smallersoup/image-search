import csv
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import torch
import torchvision.models as models
from glob import glob
from pathlib import Path
import gradio as gr
from torchvision.models import ResNet50_Weights
from reverse_image_search.tcvdb_client import TcvdbClient

# tcvdb parameters
HOST = 'lb-xxx.clb.ap-beijing.tencentclb.com'
PORT = '10000'
DB_NAME = 'image-search'
COLLECTION_NAME = 'reverse_image_search'

PASSWORD = 'xxxxx'
USERNAME = 'root'

# path to csv (column_1 indicates image path) OR a pattern of image paths
INSERT_SRC = 'reverse_image_search.csv'

# Test Image Path
TEST_IMAGE_PATH = './test/goldfish/*.JPEG'

# Initialize model
# model = models.resnet50(pretrained=True)
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Set model to eval mode
model = model.eval()
model = torch.nn.Sequential(*(list(model.children())[:-1]))


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


# Embedding: Function to extract features from an image
def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Read the image Ensure the image is read as RGB
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    feature = model(img)
    # Reshape the features to 2D
    feature = feature.view(feature.shape[0], -1)
    return feature.detach().numpy()


def display_multiple_embeddings(image_path_pattern):
    # Use glob to get all matching image paths
    image_paths = load_image(image_path_pattern)
    # Process each image and collect the results
    results = []
    for img_path in image_paths:
        feature = extract_features(img_path)
        # Convert features to a pandas DataFrame
        d = pd.DataFrame(feature)
        results.append(d)

    # Now 'results' is a list of DataFrames, one for each image.
    # You can print them or manipulate them as you wish.
    # Here we just print each one.
    for d in results:
        print(d)


def search_similar_image(path):
    for query_image in load_image(path):
        features = extract_features(query_image)
        search_res = tcvdb_client.search(features)
        # Process the search results
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
        pred = [str(Path(res[0]).resolve()) for res in search_res]
        print("Query image:", query_image)
        print("Search results:", pred)
        return pred


if __name__ == '__main__':
    # Initialize the TCVDB client
    tcvdb_client = TcvdbClient(host=HOST, port=PORT, username=USERNAME, key=PASSWORD,
                               dbName=DB_NAME, collectionName=COLLECTION_NAME, timeout=20)
    # 测试前清理环境
    # tcvdb_client.clear()
    # tcvdb_client.create_db_and_collection()

    # Display embedding result, no need for implementation
    display_multiple_embeddings(TEST_IMAGE_PATH)

    # Insert data
    # Read the CSV file to get all image paths, process each row and insert the features into the TCVDB
    # for image_path in load_image(INSERT_SRC):
    #     features = extract_features(image_path)
    #     tcvdb_client.upsert(image_path, features)

    # Search for example query image(s), process each query image and search in the TCVDB
    # search_similar_image(TEST_IMAGE_PATH)

    # webui
    iface = gr.Interface(
        fn=search_similar_image,
        inputs=gr.components.Textbox(value='test/goldfish/*.JPEG'),
        outputs=gr.Gallery(label="最终的结果图片").style(height='auto'),
        title='Tencent vector db 案例: 以图搜图',
    )
    iface.launch()
