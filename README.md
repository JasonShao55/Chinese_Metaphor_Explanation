
# CMDAG: A Chinese Metaphor Dataset with Annotated Grounds
这是一个经过精心标注、严格格式化的中文隐喻数据集（~28k），该数据集经过专家的质量检查并具有普遍性特征，适合用于大规模模型预训练等应用场景。

This is a meticulously annotated, rigorously formatted Chinese metaphor dataset (~28k) that's been expertly quality-checked and exhibits universal characteristics, making it suitable for applications such as large-scale model pre-training.

*Read this in [English](README_en.md).*

## 💹 统计 Statistics

- **Average sentence length: 59.137**
- **Number of tenor-vehicle pairs: 36002**
- **Number of total contexts: 27989**

## :sunrise: 数据集统计信息可视化 **Visualization:** 

![Tenors Word Cloud_word_cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/14e9a9e3-1d3c-4fea-bf38-f963c6d3ad18)
![Tenors (English) Word Cloud_word_cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/a507d29c-0657-47f9-b94d-d166242e3bc0)
![Vehicles Word Cloud_word_cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/5c15a5bd-1303-491d-8fec-820c6d17e2ab)
![Vehicles (English) Word Cloud_word_cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/6678e5ec-9148-4b8a-84c6-c222f4969e12)
![Grounds (Noun) Word Cloud_word_cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/c5e264e3-9d37-40c2-9d73-8836cc523ec3)
![Grounds (Noun) (English) Word Cloud_word_cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/b930a5c9-dcab-432d-8494-7b6135fdcc9f)
![Grounds (Adjective) Word Cloud_word_cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/d7ed51cb-14b8-4cbf-87e2-bbd1197740c9)
![Grounds (Adjective) (English) Word Cloud_word_cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/1f9a1248-127b-40dd-939d-38ecd2ea7e46)

## :airplane: 大模型（LLM）实验进度 Large Language Models (LLM) Experiment Progress
- [X] OpenAI Models
  - [X] ChatGPT 3.5 Turbo
  - [X] ChatGPT 4.0
- [X] ChatGLM
- [X] Baichuan
- [ ] Belle
- [ ] Tiger(After instruction tune)
- [ ] Aquila
- [ ] Claude
- [ ] CPM-Bee
- [ ] Linly
- [ ] and more...

**模型挑选规则**：
- 支持中文
- 版本尽可能新
- 可以进行CoT和Prompt Engineering

**Model Selection Criteria**
- Supports Chinese.
- Uses the latest version as much as possible.
- Can perform CoT and Prompt Engineering.

## :earth_asia: 大模型实验细节
### 实验目的 Purpose of the Experiment
证明该标注完善的中文隐喻数据集，尤其在有Ground标注信息的情况下有助于大模型的隐喻生成和喻体生成任务表现，得到每个模型在相同的六个设定下的表现结果
To demonstrate that this well-annotated Chinese metaphor dataset, especially with the presence of Ground annotations, can aid large models in metaphor generation and vehicle generation tasks. The aim is to obtain performance results of each model under the same six settings.

### 实验设定 Experimental Setup
**实验变量** **Variables**：
- 聚类（cluster）方法：2种，基于[CLS] token的embedding和基于每个单词的embedding，
  分别对应`utils\bert_cluster.py`中`cluster_and_save`（以下简称`聚类1`）和`cluster_and_save_word_embeddings`（以下简称`聚类2`）函数
- 任务类型：2种，基于Ground的CoT的隐喻生成和基于Vehicle的CoT的隐喻生成
  - 基于Ground的CoT：
    - 实验组：基于CoT, 第一步是**基于已有的聚类采样的范例，本体(Tenor)和喻体(Vehicle)，生成共性(Ground)**，第二步是基于第一步生成的共性，使用已有的本体和喻体，生成隐喻
    - 对照组：不使用CoT, 直接使用已有的**本体和喻体**，生成隐喻
  - 基于Vehicle的CoT：
    - 实验组：基于CoT, 第一步是**基于已有的聚类采样的范例，本体(Tenor)和共性(Ground)，生成喻体(Vehicle)**，第二步是基于第一步生成的喻体，使用已有的本体和共性，生成隐喻
    - 对照组：不使用CoT, 直接使用已有的**本体和共性**，生成隐喻
    
因此，每个模型的实验组数是6组：基于Ground的CoT（聚类1，聚类2），基于Ground的CoT对照组，基于Vehicle的CoT（聚类1，聚类2），基于Vehicle的CoT对照组

**已有的聚类采样的范例**：指本体，喻体，共性和隐喻完善的少数样例(20条)，基于聚类1和聚类2从原数据集的训练集（train_data.json）中采样得到，保存在`Data\Selected Data`中，聚类1对应`train_clustered_sampled_20.json`, 聚类2对应`train_clustered_sampled_embeddings_20.json`，在Prompt模型时作为第一步CoT的共同输入（如果模型有记忆能力，可以作为对话一开始的一个输入）
  
**实验数据**：
`Data\Splited Data`是本数据集分割后的数据：
- train_data: 按id排序的前80%数据，LLM实验的数据来源
- val_data:按id排序, train_data之后的10%数据
- test_data:按id排序, val_data之后的10%数据
- `train_data_sampled_200.json`:从train_data随机选择，清洗再随机挑选后的**统一用于大模型实验输入的数据**

**实验步骤**：
- 示例代码：
基于OpenAI GPT-4完成实验的代码在`Code\OpenAI Sample Code`，CoT具体的Prompt内容需要为中文，可基于模型输出效果调整Prompt的措辞，但要求保持信息的完整性，不得删除或添加关键信息，改用不同模型实验时尽可能保持统一。
- 示例结果：
已完成的模型的实验结果在`Experiment_Results/Model_Name`目录下，每个模型的结果是6个json文件，记录上述6种设定下模型的prompt和输出等信息，**要求格式， tenor，每条语料顺序等严格一致**，建议每条语料内按输入在前，输出在后的顺序排列，比如基于Ground的CoT的任务的结果截取如下：
```json
    {
        "tenor": "语言",
        "vehicle": "刀",
        "ground": "XXX",
        "metaphor": "XXXX"
    },
```
相应地，基于Vehicle的CoT的任务的结果截取如下：
```json
    {
        "tenor": "语言",
        "ground": "无情的伤害", 
        "vehicle": "XXX",
        "metaphor": "XXX"
    },
```

## 📌 引用  Reference

如果使用本项目的代码、数据或模型，请引用本项目。

If you were to use this project's code, dataset, or model please refer to it as following.

```
@misc{cn_metaphor_yujie,
  author = {Yujie Shao*, Xinrong Yao*, Ge Zhang, Linyuan Zhang, Zijie Wang, Yifan Liu, Yaoyao Wu, Yunji Liu, Jie Fu+, Shi Wang+},
  title = {CMDAG: A Chinese Metaphor Dataset with Annotated Grounds},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JasonShao55/Chinese_Metaphor_Explanation}},
}
```

