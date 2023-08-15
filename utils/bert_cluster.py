import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch
import json
from tqdm import tqdm
import os
  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 选择设备，如果可用的话使用 GPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 使用BERT embeddings和K-means聚类
def cluster_and_save(input_file, output_file, n_clusters, batch_size=100):
    # 预训练模型的名称
    model_name = 'bert-base-chinese'

    # 初始化tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)  # 将模型移动到指定设备上

    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 初始化一个空的numpy数组来保存所有的embeddings
    all_embeddings = np.empty((0, model.config.hidden_size))

    # 将数据分为多个批次，然后一次处理一个批次
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i + batch_size]
        inputs = tokenizer(list(batch['context']+' '+batch['tenor']+' '+batch['vehicle']+' '+batch['ground']), padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = inputs.to(device)  # 将输入数据移动到指定设备上

        # 获取文本的BERT embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # 将计算结果移动回 CPU

        # 将这个批次的embeddings添加到all_embeddings中
        all_embeddings = np.vstack((all_embeddings, embeddings))

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_embeddings)

    # 添加聚类结果到数据中
    df['cluster_id'] = kmeans.labels_

    # 按cluster_id排序
    df = df.sort_values(by=['cluster_id'])

    # 写入新的json文件
    df['metaphor_id']=df['metaphor_id'].astype(int)
    output_data = df.to_dict('records')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4, separators=(",", ": "))


def cluster_and_save_word_embeddings(input_file, output_file, n_clusters, batch_size=100):
    # 预训练模型的名称
    model_name = 'bert-base-chinese'

    # 初始化tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)  # 将模型移动到指定设备上

    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 初始化一个空的numpy数组来保存所有的embeddings
    all_embeddings = np.empty((0, model.config.hidden_size))

    # 将数据分为多个批次，然后一次处理一个批次
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i + batch_size]
        inputs = tokenizer(list(batch['context']+' '+batch['tenor']+' '+batch['vehicle']+' '+batch['ground']), padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = inputs.to(device)  # 将输入数据移动到指定设备上

        # 获取文本的BERT embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # embeddings now holds word-level embeddings for each input token
            embeddings = outputs.last_hidden_state.cpu().numpy()  # 将计算结果移动回 CPU

        # 将这个批次的embeddings添加到all_embeddings中
        for embedding in embeddings:
            all_embeddings = np.vstack((all_embeddings, embedding))

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_embeddings)

    # 添加聚类结果到数据中
    df['cluster_id'] = kmeans.labels_

    # 按cluster_id排序
    df = df.sort_values(by=['cluster_id'])

    # 写入新的json文件
    df['metaphor_id']=df['metaphor_id'].astype(int)
    output_data = df.to_dict('records')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4, separators=(",", ": "))


# 每类随机抽取一个样本
def sample_and_save(input_file, output_file, random_seed=0):
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 对每个cluster_id分组，然后在每个组中随机选取一个样本
    sampled_df = df.groupby('cluster_id').sample(n=1, random_state=random_seed)

    # 写入新的json文件
    output_data = sampled_df.to_dict('records')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4, separators=(",", ": "))



if __name__ == '__main__':
    '''
    # 使用聚类函数
    cluster_and_save('/home/zhangge/yujie/data/train_data.json', '/home/zhangge/yujie/data/train_clustered.json', 20)
    # 使用采样函数
    sample_and_save('/home/zhangge/yujie/data/train_clustered.json', '/home/zhangge/yujie/data/train_clustered_sampled.json')
    '''

    '''
    # 使用聚类函数
    cluster_and_save('/home/zhangge/yujie/data/train_data.json', '/home/zhangge/yujie/data/train_clustered_embeddings.json', 20)
    # 使用采样函数
    sample_and_save('/home/zhangge/yujie/data/train_clustered_embeddings.json', '/home/zhangge/yujie/data/train_clustered_sampled_embeddings.json')
    '''

    # 改seed 避开问题数据
    sample_and_save('/home/zhangge/yujie/data/train_clustered_embeddings.json', '/home/zhangge/yujie/data/train_clustered_sampled_embeddings.json',random_seed=1)
    sample_and_save('/home/zhangge/yujie/data/train_clustered.json', '/home/zhangge/yujie/data/train_clustered_sampled.json',random_seed=1)