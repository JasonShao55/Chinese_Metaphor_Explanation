# CMDAG: A Chinese Metaphor Dataset with Annotated Grounds

This dataset provides a meticulously annotated and rigorously formatted collection of Chinese metaphors (~28k entries). The dataset has undergone expert quality checks and demonstrates universal characteristics. This makes it especially suitable for applications such as pre-training large-scale models.

*For other language versions, refer to [Chinese](README.md).*

## 💹 Statistics

- 📏 Average sentence length: 59.137
- 🚀 Number of tenor-vehicle pairs: 36002
- 📄 Total contexts: 27989

## :sunrise: Visualization of Dataset Statistics

Various word clouds depicting the frequency and variety of terms used in the dataset:

![Tenors Word Cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/14e9a9e3-1d3c-4fea-bf38-f963c6d3ad18)
![Tenors Word Cloud in English](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/a507d29c-0657-47f9-b94d-d166242e3bc0)
![Vehicles Word Cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/5c15a5bd-1303-491d-8fec-820c6d17e2ab)
![Vehicles Word Cloud in English](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/6678e5ec-9148-4b8a-84c6-c222f4969e12)
![Grounds (Noun) Word Cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/c5e264e3-9d37-40c2-9d73-8836cc523ec3)
![Grounds (Noun) Word Cloud in English](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/b930a5c9-dcab-432d-8494-7b6135fdcc9f)
![Grounds (Adjective) Word Cloud](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/d7ed51cb-14b8-4cbf-87e2-bbd1197740c9)
![Grounds (Adjective) Word Cloud in English](https://github.com/JasonShao55/Chinese_Metaphor_Explanation/assets/61415289/1f9a1248-127b-40dd-939d-38ecd2ea7e46)

## :airplane: Large Language Models (LLM) Experiment Progress

Progress status of various experiments with different models:

- [X] OpenAI Models 🌐
  - [X] ChatGPT 3.5 Turbo 🤖
  - [X] ChatGPT 4.0 🤖
- [X] ChatGLM 🤓
- [X] Baichuan 🌊
- [ ] Belle 🛎️
- [ ] Tiger (Post-instruction tune) 🐅
- [ ] Aquila 🦅
- [ ] Claude 🎨
- [ ] CPM-Bee 🐝
- [ ] Linly 🎼
- [ ] and more... 🔍

💼 **Model Selection Criteria**:
- 🌏 Supports Chinese
- 🔄 Embraces the latest versions
- 🧪 Capable of CoT and Prompt Engineering tasks

## 📖 Experiment Details

### 🎯 Purpose of the Experiment

The main aim is to showcase that this well-curated dataset, enriched with Ground annotations, can significantly assist large-scale models in metaphor and vehicle generation tasks. The evaluations for each model are consistent across six standardized settings.

### 🛠️ Experimental Setup

**📊 Variables**:

- 📌 Clustering methods: There are two, one based on the [CLS] token embedding and the other on individual word embeddings. These correspond to the functions `cluster_and_save` (referred to as `Cluster 1`) and `cluster_and_save_word_embeddings` (referred to as `Cluster 2`) in `utils\bert_cluster.py`.

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

## 🖋️ Reference

If you utilize the code, data, or any models from this project, please cite the project as follows:
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