# -*- coding: UTF-8 -*-
import os
import zipfile
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
import re
from PIL import Image



plt.rcParams['font.sans-serif'] = ['simhei']  # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 设置在使用unicode时能够显示负号
sns.set(font='simhei')  # 设置字体为SimHei

# base_path = r'D:\OneDrive\Learning\NLP Metaphor Dataset\Chinese_Metaphor_Explanation\annotated_ subcorpora'
# tem_path = r'D:\OneDrive\Learning\NLP Metaphor Dataset\annotated_subcorpora_tem'
# final_path = r'D:\OneDrive\Learning\NLP Metaphor Dataset\Chinese_Metaphor_Explanation'
corpus_path = 'annotated_corpora.json'

# 批量处理json文本
# 注意格式：创建一个list，再将每个改过的item(dict)加入到list中，dump这个list到json文件中
# 一个文件里有多个数据集(jsonl文本) 把他们进行一些处理后复制到另一个路径中：
def copy_to_final(input_file_path, output_file_path):
    # 指定需要保留的字段
    fields_to_keep = ['metaphor_id', 'context', 'final_tenor', 'final_vehicle', 'final_ground']
    for root, dirs, files in os.walk(input_file_path):
        for file in files:
            if file.endswith('.json'):
                # 打开输入文件, input_file_path, file进行路径合并
                with open(os.path.join(input_file_path, file), 'r', encoding='utf-8') as input_file:
                    # 打开输出文件
                    with open(os.path.join(output_file_path, file), 'w', encoding='utf-8') as output_file:
                        try:
                            # json文件必须用json.load()方法读取才会被标准化
                            input_file = json.load(input_file)
                            # 遍历输入文件的每一行
                            new_data = []
                            for item in input_file:
                                # 仅保留需要的字段
                                item = {key: item[key] for key in fields_to_keep if key in item}
                                new_data.append(item)
                                '''
                                # 修改key的名字
                                for old_key, new_key in key_name_change.items():
                                    if old_key in filtered_data:
                                        filtered_data[new_key] = key_name_change.pop(old_key)
                                '''
                                # 将处理后的数据写入到输出文件中
                            output_file.write(json.dumps(new_data, ensure_ascii=False, indent=2,
                                                         separators=(",", ": ")))  # 别转义中文
                        except Exception as error:
                            print(error)
                            print(output_file_path)


def key_name_changes(input_file_path, output_file_path):
    key_name_change = {'final_tenor': 'tenor', 'final_vehicle': 'vehicle', 'final_ground': 'ground'}
    for root, dirs, files in os.walk(input_file_path):
        for file in files:
            if file.endswith('.json'):
                # 打开输入文件, input_file_path, file进行路径合并
                with open(os.path.join(input_file_path, file), 'r', encoding='utf-8') as input_file:
                    # 打开输出文件
                    with open(os.path.join(output_file_path, file), 'w', encoding='utf-8') as output_file:
                        # json文件必须用json.load()方法读取才会被标准化
                        new_data = []
                        input_file = json.load(input_file)
                        # 遍历输入文件的每一行
                        for item in input_file:
                            # 修改key的名字
                            for old_key, new_key in key_name_change.items():
                                if old_key in item:
                                    item[new_key] = item.pop(old_key)
                            new_data.append(item)
                            # 将处理后的数据写入到输出文件中
                        output_file.write(json.dumps(new_data, ensure_ascii=False, indent=2,
                                                     separators=(",", ": ")))  # 别转义中文


def create_wordcloud(words, title, language):
    # 形状与颜色设置
    mask_image = np.array(Image.open('cloud_image_3.png'))
    color_palatte = mcolors.ListedColormap(['#f2c61f', '#5cbd72', '#564f8a', '#33c1bb', '#dc54a0',
                                            '#e48c6a', '#000000', '#dc6a69', '#3b83c0', '#808080'])
    # 中文词云
    if language == 'zh':
        wordcloud = WordCloud(font_path='SourceHanSerif/SourceHanSerifK-Light.otf',
                              mask=mask_image,
                              prefer_horizontal=1,
                              width=8000, height=4000,
                              colormap=color_palatte,
                              background_color='white',
                              collocations=False).generate(" ".join(words))
    # 英文词云
    elif language == 'en':
        wordcloud = WordCloud(mask=mask_image,
                              prefer_horizontal=1,
                              width=8000, height=4000,
                              colormap=color_palatte,
                              background_color='white',
                              collocations=False).generate(" ".join(words))

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis("off")
    plt.savefig(title + '_word_cloud.svg')
    # plt.savefig(title+'_word_cloud.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


# 绘制箱型图
def create_boxplot_lengths_of_items(df_grounds, df_vehicles, df_tenors):
    # 颜色设置
    color_palatte = sns.color_palette(['#f0bb41', '#c52a20', '#508ab2'])

    # 创建一个箱形图显示ground, vehicle, tenor的长度分布
    df_grounds['ground_len'] = df_grounds['ground'].apply(len)
    df_vehicles['vehicle_len'] = df_vehicles['vehicle'].apply(len)
    df_tenors['tenor_len'] = df_tenors['tenor'].apply(len)
    df_combined = pd.concat([df_grounds['ground_len'], df_vehicles['vehicle_len'], df_tenors['tenor_len']], axis=1)

    plt.figure(figsize=(10, 5))
    # sns.boxplot(data=df[['ground_len', 'vehicle_len', 'tenor_len']])
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.boxplot(data=df_combined, palette=color_palatte)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    new_labels = ['Ground Length', 'Vehicle Length', 'Tenor Length']
    plt.gca().set_xticklabels(new_labels)
    plt.title('Distribution of Lengths for Ground, Vehicle, and Tenor')
    plt.ylabel('Length')
    # plt.savefig('box_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('box_plot_lengths_of_items.svg')
    plt.show()


def most_frequent_items(df_grounds, df_vehicles, df_tenors):
    # 字体与颜色设置
    font_prop = fm.FontProperties(fname='SourceHanSerif/SourceHanSerifK-Light.otf')
    color_palatte = sns.color_palette(['#f0bb41', '#c52a20', '#508ab2'])

    # 计算 ground, vehicle, tenor的频率
    ground_freq = df_grounds.groupby('ground').size().sort_values(ascending=False).reset_index()
    vehicle_freq = df_vehicles.groupby('vehicle').size().sort_values(ascending=False).reset_index()
    tenor_freq = df_tenors.groupby('tenor').size().sort_values(ascending=False).reset_index()

    top_list = []
    frequency_list = []
    category_list = []

    categories = ['Ground', 'Vehicle', 'Tenor']
    freq_dfs = [ground_freq, vehicle_freq, tenor_freq]
    for i in range(3):
        category = categories[i]
        freq_df = freq_dfs[i]
        for j in range(3):
            top = freq_df.iloc[j].values[0]
            frequency = freq_df.iloc[j].values[1]

            top_list.append(top)
            frequency_list.append(frequency)
            category_list.append(category)

    # 构建一个 DataFrame 包含统计数据
    stats_data = {
        "category": category_list,
        "top": top_list,
        "frequency": frequency_list
    }
    stats_df = pd.DataFrame(stats_data)
    # 创建一个图形的网格
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    plt.xticks(fontproperties=font_prop)

    # 条形图显示 "ground", "vehicle" 和 "tenor" 的频率
    sns.barplot(x="top", y="frequency", hue='category', dodge=False,
                data=stats_df, ax=axes, palette=color_palatte)

    axes.set_xlabel("Top Labels")
    axes.set_ylabel("Label Frequency")
    axes.set_title("Frequency of Top Ground, Vehicle, and Tenor Labels")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #     new_labels = ['Ground', 'Vehicle', 'Tenor']
    #     plt.gca().set_xticklabels(new_labels)
    # 保存和显示图形
    plt.legend(loc='upper right')
    plt.tight_layout()
    # plt.savefig('bar_graph.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('bar_graph_top_items.svg')
    plt.show()


# Top10 Ground, Ground的名词，Ground的形容词
def top10_ground(grounds):
    # 字体设置
    font_prop = fm.FontProperties(fname='SourceHanSerif/SourceHanSerifK-Light.otf')

    adjectives = []
    nouns = []
    for ground in grounds:
        words = ground.split('的')
        if len(words) == 2:
            adjectives.append(words[0])
            nouns.append(words[1])
        else:
            print('出现格式有问题的ground：', ground)

    # 计算形容词和名词的频率
    adjective_counts = Counter(adjectives)
    noun_counts = Counter(nouns)
    ground_counts = Counter(grounds)

    # 获取最常见的10个形容词和名词
    top_10_adjectives = adjective_counts.most_common(10)
    top_10_nouns = noun_counts.most_common(10)
    top_10_grounds = ground_counts.most_common(10)

    # 创建DataFrame
    df_adjectives = pd.DataFrame(top_10_adjectives, columns=['Adjective', 'Count'])
    df_nouns = pd.DataFrame(top_10_nouns, columns=['Noun', 'Count'])
    df_grounds = pd.DataFrame(top_10_grounds, columns=['Ground', 'Count'])

    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(10, 16))

    # 第一个图：显示Top10最常见的ground
    sns.barplot(x="Count", y="Ground", data=df_grounds, ax=axes[0])
    axes[0].set_title("Top 10 Most Common Ground Phrases")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontproperties=font_prop)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set_xlabel('Frequency')

    # 第二个图：显示Top10最常见的形容词
    sns.barplot(x="Count", y="Adjective", data=df_adjectives, ax=axes[1])
    axes[1].set_title("Top 10 Most Common Adjectives in Ground")
    axes[1].set_yticklabels(axes[1].get_yticklabels(), fontproperties=font_prop)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_xlabel('Frequency')

    # 第三个图：显示Top10最常见的名词
    sns.barplot(x="Count", y="Noun", data=df_nouns, ax=axes[2])
    axes[2].set_title("Top 10 Most Common Nouns in Ground")
    axes[2].set_yticklabels(axes[2].get_yticklabels(), fontproperties=font_prop)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].set_xlabel('Frequency')

    # 保存和显示图形
    plt.tight_layout()
    plt.savefig('bar_graph_top_ground_items.svg')
    plt.show()


def stats_visualization(corpus_path):
    # 读取数据集
    # data = []
    # for root, dirs, files in os.walk(input_file_path):
    #     for file in files:
    #         if file.endswith('.json'):
    #             with open(os.path.join(input_file_path, file), 'r', encoding='utf-8') as file:
    #                 data.extend(json.load(file))
    with open(corpus_path, 'r') as file:
        data = json.load(file)
    # 初始化统计变量
    total_length = 0
    total_metaphors = len(data)
    grounds = []
    vehicles = []
    tenors = []
    pair_number = 0
    # 处理数据集
    for item in tqdm(data, desc='Processing data'):
        total_length += len(item['context'])
        # 按标注标点规则拆解字符串
        real_tenor = item['tenor'].split('；')
        tenors += real_tenor
        real_vehicle = re.split('；|，', item['vehicle'])
        vehicles += real_vehicle
        grounds += re.split('；|，|、', item['ground'])

        pair_number += len(real_tenor) * len(real_vehicle)

    # 计算和显示平均句子长度
    average_length = total_length / total_metaphors
    print(f'Average sentence length: {average_length}')

    # 计算和显示有多少对本喻体：
    print(f'Number of tenor-vehicle pairs: {pair_number}')

    print(f'Total Metaphors: {total_metaphors}')

    # 创建DataFrame，三者分割前可以放在同一个dataframe中
    # df = pd.DataFrame({'ground': grounds, 'vehicle': vehicles, 'tenor': tenors})
    # 创建DataFrame
    df_grounds = pd.DataFrame({'ground': grounds})
    df_vehicles = pd.DataFrame({'vehicle': vehicles})
    df_tenors = pd.DataFrame({'tenor': tenors})

    adjectives = []
    nouns = []
    for ground in grounds:
        words = ground.split('的')
        if len(words) == 2:
            adjectives.append(words[0])
            nouns.append(words[1])
        else:
            print('出现格式有问题的ground：', ground)

    grounds_eng = pd.read_excel('ground_en.xlsx')
    adjectives_eng = list(grounds_eng['adjective'].astype(str).str.lower())
    nouns_eng = list(grounds_eng['noun'].astype(str).str.lower())
    vehicles_eng = list(pd.read_excel('vehicle_en.xlsx')['vehicle'].astype(str).str.lower())
    tenors_eng = list(pd.read_excel('tenor_en.xlsx')['tenor'].astype(str).str.lower())

    # 显示词云
    print("——————————开始绘制词云——————————")
    create_wordcloud(grounds, 'Grounds Word Cloud', 'zh')
    create_wordcloud(adjectives, 'Grounds (Adjective) Word Cloud', 'zh')
    create_wordcloud(nouns, 'Grounds (Noun) Word Cloud', 'zh')
    create_wordcloud(vehicles, 'Vehicles Word Cloud', 'zh')
    create_wordcloud(tenors, 'Tenors Word Cloud', 'zh')

    create_wordcloud(adjectives_eng, 'Grounds (Adjective) (English) Word Cloud', 'en')
    create_wordcloud(nouns_eng, 'Grounds (Noun) (English) Word Cloud', 'en')
    create_wordcloud(vehicles_eng, 'Vehicles (English) Word Cloud', 'en')
    create_wordcloud(tenors_eng, 'Tenors (English) Word Cloud', 'en')

    # # 显示数据集的统计信息
    print(df_grounds.describe())
    print(df_vehicles.describe())
    print(df_tenors.describe())

    create_boxplot_lengths_of_items(df_grounds, df_vehicles, df_tenors)
    most_frequent_items(df_grounds, df_vehicles, df_tenors)
    top10_ground(grounds)


if __name__ == '__main__':
    # copy_to_final(base_path, tem_path)
    # key_name_changes(tem_path, final_path)
    stats_visualization(corpus_path)
