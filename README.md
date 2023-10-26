# CMDAG: A Chinese Metaphor Dataset with Annotated Grounds

This dataset provides a meticulously annotated and rigorously formatted collection of Chinese metaphors (~28k entries). The dataset has undergone expert quality checks and demonstrates universal characteristics. This makes it especially suitable for applications such as pre-training large-scale models.

***🔍 Read this in [Chinese](README_trans.md).***


## 💹 Statistics

- 📏 Average sentence length: 59.137
- 🚀 Number of tenor-vehicle pairs: 36002
- 📄 Total contexts: 27989

## :airplane: Large Language Models (LLM) Experiment Progress

Progress status of various experiments with different models:

- [X] OpenAI Models🌐
  - [X] ChatGPT 3.5 Turbo🤖
  - [X] ChatGPT 4.0 🤖
- [X] ChatGLM🤓
- [X] Baichuan🌊
- [X] Belle🛎️
- [X] Baidu ERNIE🦅
- [X] TigerBot🐅
  - [X] TigerBot-7B-Chat🐾
  - [X] TigerBot-13B-Chat🐾
- [X] Chinese-alpaca🦙
  - [X] Chinese-alpaca-7B🏮
  - [X] Chinese-alpaca-13B🏮
  - [X] Chinese-alpaca-33B🏮
- [X] Chinese-alpaca-2🦙
  - [X] Chinese-alpaca-2-7B🎏
  - [X] Chinese-alpaca-2-13B🎏
- [X] chinese-llama2-linly🦙
  - [X] chinese-llama2-linly-7B🍃
  - [X] chinese-llama2-linly-13B🍃
- [X] Qwen-7B-Chat🌟
- [X] Ziya-LLaMA-13B🌌


💼**Model Selection Criteria：**
- 🌏 Supports Chinese
- 🔄 Uses the latest version as much as possible
- 🧪 Engineering Can perform CoT and Prompt Engineering

## 📜  Human evaluation progresss

- [X] OpenAI Models🌐
  - [X] ChatGPT 3.5 Turbo🤖
  - [X] ChatGPT 4.0 🤖
- [X] ChatGLM🤓
- [X] Baichuan🌊
- [X] Belle🛎️
- [X] Baidu ERNIE🦅
- [X] TigerBot🐅
  - [X] TigerBot-7B-Chat🐾
  - [X] TigerBot-13B-Chat🐾
- [X] Chinese-alpaca🦙
  - [X] Chinese-alpaca-7B🏮
  - [X] Chinese-alpaca-13B🏮
  - [X] Chinese-alpaca-33B🏮
- [X] Chinese-alpaca-2🦙
  - [X] Chinese-alpaca-2-7B🎏
  - [X] Chinese-alpaca-2-13B🎏
- [X] chinese-llama2-linly🦙
  - [X] chinese-llama2-linly-7B🍃
  - [X] chinese-llama2-linly-13B🍃
- [X] Qwen-7B-Chat🌟
- [X] Ziya-LLaMA-13B🌌

💼 **Model Selection Criteria**:
- 🌏 Supports Chinese
- 🔄 Embraces the latest versions
- 🧪 Capable of CoT and Prompt Engineering tasks

## 📖 Experiment Details

### 🎯 Purpose of the Experiment

The main aim is to showcase that this well-curated dataset, enriched with Ground annotations, can significantly assist large-scale models in metaphor and vehicle generation tasks. The evaluations for each model are consistent across six standardized settings.

### 🛠️ Experimental Setup

**📊 Variables**:

- 📌 Clustering methods: There are two, one based on the [CLS] token embedding (sentence-level clustering) and the other on individual word embeddings （word-level clustering）. These correspond to the functions `cluster_and_save` (referred to as `Cluster 1`) and `cluster_and_save_word_embeddings` (referred to as `Cluster 2`) in `utils\bert_cluster.py`.

- 🎩 Task types: There are two main tasks: metaphor generation based on Ground CoT and metaphor generation based on Vehicle CoT.

Specifics regarding the tasks are as outlined:

1. Ground-based CoT:
   - Experimental group: Starting with **samples from existing clusters**, the first step involves generating the shared Ground using tenors and vehicles. Metaphors are then generated using this derived Ground alongside the existing tenor and vehicle.
   - Control group: Without using CoT, directly create metaphors using existing tenors and vehicles.

2. Vehicle-based CoT:
   - Experimental group: The first step involves **generation of the vehicle** using the tenor and Ground sampled from existing clusters. The subsequent step revolves around generating metaphors with the derived vehicle along with existing tenor and Ground.
   - Control group: Without CoT, directly form metaphors using the tenors and Ground.

🗄️ **Sample Clusters**: These are a select few examples (20 entries) that have been well-annotated with tenor, vehicle, Ground, and metaphor. They've been sampled from the training set of the original dataset based on the two clustering methods mentioned above. These samples are stored under `Data\Selected Data`.

🔎 **Data for Experiments**: The split data for the dataset resides under `Data\Splited Data`. The directory contains:

- `train_data`: The first 80% of data sorted by ID, used as the primary source for LLM experiments.
- `val_data`: The subsequent 10% after `train_data`, sorted by ID.
- `test_data`: The following 10% after `val_data`, sorted by ID.
- 📂 `train_data_sampled_200.json`: This is a cleansed and selectively sampled subset from `train_data`, chosen specifically for the large model experiments.

📜 **Procedure**:
- 🖥️ **Code Samples**: The sample code to run experiments on OpenAI GPT-4 can be found under `Code\OpenAI Sample Code`. The specific CoT Prompt needs to be in Chinese. Prompts can be adjusted based on model outputs, but must remain informationally intact.
- 📉 **Sample Results**: The results of the experiments with completed models are stored in the `Experiment_Results/Model_Name` directory. Each model's results are represented as six JSON files. They capture the prompts, outputs, and other data for the model across the six settings.

## :sunrise: Visualization of Dataset Statistics

Various word clouds depicting the frequency and variety of terms used in the dataset:

![Tenors Word Cloud_word_cloud](Visualization/Tenors%20Word%20Cloud_word_cloud.png)
![Tenors (English) Word Cloud_word_cloud](Visualization/Tenors%20(English)%20Word%20Cloud_word_cloud.png)
![Vehicles Word Cloud_word_cloud](Visualization/Vehicles%20Word%20Cloud_word_cloud.png)
![Vehicles (English) Word Cloud_word_cloud](Visualization/Vehicles%20(English)%20Word%20Cloud_word_cloud.png)
![Grounds (Noun) Word Cloud_word_cloud](Visualization/Grounds%20(Noun)%20Word%20Cloud_word_cloud.png)
![Grounds (Noun) (English) Word Cloud_word_cloud](Visualization/Grounds%20(Noun)%20(English)%20Word%20Cloud_word_cloud.png)
![Grounds (Adjective) Word Cloud_word_cloud](Visualization/Grounds%20(Adjective)%20Word%20Cloud_word_cloud.png)
![Grounds (Adjective) (English) Word Cloud_word_cloud](Visualization/Grounds%20(Adjective)%20(English)%20Word%20Cloud_word_cloud.png)


## 🖋️ Reference

If you utilize the code, data, or any models from this project, please cite the project as follows:
```
@inproceedings{cn_metaphor_yujie,
    author    = {Yujie Shao* and Xinrong Yao* and Xingwei Qu* and Chenghua Lin and Shi Wang and Wenhao Huang and Ge Zhang+ and Jie Fu+},
    title     = {CMDAG: A Chinese Metaphor Dataset with Annotated Grounds as CoT for Boosting Metaphor Generation},
    year      = {2024},
    booktitle = {Proceedings of the LREC-COLING 2024},
    url       = {https://anonymous.4open.science/r/Chinese_Metaphor_Explanation-63F2},
}
```
