import json     # 用于加载 JSON
import os       # 文件操作
import random   # 随机选择数据集
import requests  # 通过 HTTP 请求从 coco_url 下载图片

# 设置存储路径
des_train = './transformer_cnn/database/images_train'
os.makedirs(des_train, exist_ok=True)  # 创建目标文件夹用于存放训练集图片

des_val = './transformer_cnn/database/images_val'
os.makedirs(des_val, exist_ok=True)      # 创建目标文件夹用于存放验证集图片 

des_test = './transformer_cnn/database/images_test'
os.makedirs(des_test, exist_ok=True)     # 创建目标文件夹用于存放测试集图片

# 加载数据
with open('./transformer_cnn/annotations/captions_train2017.json', 'r') as f:
    train_data = json.load(f)       # 训练集+测试集数据
with open('./transformer_cnn/annotations/captions_val2017.json', 'r') as f:
    val_data = json.load(f)         # 验证集数据

train_images = train_data['images']
val_images = val_data['images']

# 随机抽取数据
select_t_images = random.sample(train_images, 90)
select_train_images = select_t_images[:80]
select_test_images = select_t_images[80:]
select_v_images = random.sample(val_images, 10)

# 创建一个文本-图像对的列表
image_caption_pairs = []

# 下载训练集图片并记录图片-文本对
for img in select_train_images:
    img_url = img.get('coco_url')  # 获取 coco_url
    if not img_url:  # 检查 image_url 是否为空
        print("图片", img['file_name'], "没有可用的 URL, 跳过")
        continue
    # 设置下载路径
    img_path = os.path.join(des_train, img['file_name'])
    
    # 找到对应的文本描述
    img_id = img['id']
    captions = [caption['caption'] for caption in train_data['annotations'] if caption['image_id'] == img_id]
    
    try:
        # 下载图片
        response = requests.get(img_url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 保存图片
        with open(img_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)  # 将读取的内容写入文件
        
        print("成功下载图片", img['file_name'])
        
        # 记录图片-文本对
        for caption in captions:
            image_caption_pairs.append({'image_path': img_path, 'caption': caption})

    except requests.exceptions.RequestException as e:
        print("下载图片", {img['file_name']}," 时出错：", {e})

# 下载测试集图片并记录图片-文本对
for img in select_test_images:
    img_url = img.get('coco_url')  # 获取 coco_url
    if not img_url:  # 检查 image_url 是否为空
        print("图片", img['file_name'], "没有可用的 URL, 跳过")
        continue
    # 设置下载路径
    img_path = os.path.join(des_test, img['file_name'])
    
    # 找到对应的文本描述
    img_id = img['id']
    captions = [caption['caption'] for caption in train_data['annotations'] if caption['image_id'] == img_id]

    try:
        # 下载图片
        response = requests.get(img_url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 保存图片
        with open(img_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)  # 将读取的内容写入文件
        
        print("成功下载图片", img['file_name'])
        
        # 记录图片-文本对
        for caption in captions:
            image_caption_pairs.append({'image_path': img_path, 'caption': caption})

    except requests.exceptions.RequestException as e:
        print("下载图片", {img['file_name']}," 时出错：", {e})

# 下载验证集图片并记录图片-文本对
for img in select_v_images:
    img_url = img.get('coco_url')  # 获取 coco_url
    if not img_url:  # 检查 image_url 是否为空
        print("图片", img['file_name'], "没有可用的 URL, 跳过")
        continue
    # 设置下载路径
    img_path = os.path.join(des_val, img['file_name'])
    
    # 找到对应的文本描述
    img_id = img['id']
    captions = [caption['caption'] for caption in val_data['annotations'] if caption['image_id'] == img_id]

    try:
        # 下载图片
        response = requests.get(img_url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 保存图片
        with open(img_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)  # 将读取的内容写入文件
        
        print("成功下载图片", img['file_name'])
        
        # 记录图片-文本对
        for caption in captions:
            image_caption_pairs.append({'image_path': img_path, 'caption': caption})

    except requests.exceptions.RequestException as e:
        print("下载图片", {img['file_name']}," 时出错：", {e})

# 保存图片-文本对数据集到 JSON 文件
with open('./transformer_cnn/database/image_caption_pairs.json', 'w') as f:
    json.dump(image_caption_pairs, f)

print("下载完成，图片-文本数据集已保存！")
