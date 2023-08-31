import pandas as pd
import os
import json
import re
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer

data_path = '/home/xryao/CMDAG/data'
model_path = '/pool001/xryao/'
# data_path = '/Users/davidyao/Documents/CMDAG/CMDAG/data'
device = torch.device('cpu')


def BELLE_Task1(CoT, sample_data, subset, output_file):
    results = []
    with open(os.path.join(data_path, 'train_data_sampled_200.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)[subset * 20:(subset + 1) * 20]

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "BELLE-7B-2M"))
    model = AutoModelForCausalLM.from_pretrained(os.path.join(model_path, "BELLE-7B-2M"))

    with open(os.path.join(data_path, sample_data), 'r', encoding='utf-8') as f:
        sampled_metaphors = json.load(f)

    grounds = ['{}作为本体（tenor）和 {}作为喻体（vehicle）构成隐喻时，'.format(m['tenor'], m['vehicle']) +
               '它们的共性（ground）可以是：{}'.format(m['ground']) for m in sampled_metaphors]
    initial_prompt = 'Human: 众所周知，隐喻（metaphor）的本体和喻体存在共性(ground)，诸如由下述本体（tenor）和喻体（vehicle）构建隐喻时，' + \
                     '它们的共性可以是以下这些：' + "，".join(grounds)

    for entity in tqdm(data, total=len(data)):
        tenor = entity['tenor']
        vehicle = entity['vehicle']
        ground = ''
        metaphor = ''
        # print(tenor, vehicle)

        if CoT:
            prompt = "请参考我给你展示的例子的内容和格式，直接给出{}和{}".format(tenor, vehicle) + \
                     "的共性（ground），最好是一个或多个'形容词+的+名词'的短语的形式，不需要多加任何其他的解释或说明。\n\nAssistant: "

            trials = 0
            while ground == '' and trials < 5:
                trials += 1

                print('PROMPT: \n {}'.format(initial_prompt + "\n\n" + prompt))
                input_ids = tokenizer(initial_prompt + "\n\n" + prompt, return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_new_tokens=100,
                                         do_sample=True, top_k=30, top_p=0.85,
                                         temperature=0.5, repetition_penalty=1.2,
                                         eos_token_id=2, bos_token_id=1, pad_token_id=0)

                response = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0].strip()
                response = response.replace(initial_prompt + "\n\n" + prompt, "")

                print('RESPONSE: \n {}'.format(response))
                response = response.replace('形容词+的+名词', '').replace('的短语', '').strip()
                if len(response) > 3:
                    ground = response.strip()

            prompt = "基于你给出的共性：“{}”，请写一个简短准确的含有关于{}和{}的隐喻（metaphor）的中文句子。\n\nAssistant: ".format(
                ground, tenor, vehicle)

            trials = 0
            while metaphor == '' and trials < 5:
                trials += 1

                print('PROMPT: \n {}'.format(initial_prompt + "\n\n" + prompt))
                input_ids = tokenizer(initial_prompt + "\n\n" + prompt, return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_new_tokens=100,
                                         do_sample=True, top_k=30, top_p=0.85,
                                         temperature=0.5, repetition_penalty=1.2,
                                         eos_token_id=2, bos_token_id=1, pad_token_id=0)

                response = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0].strip()
                response = response.replace(initial_prompt + "\n\n" + prompt, "")

                print('RESPONSE: \n {}'.format(response))
                if '共性' not in response:
                    metaphor = response.strip()
        else:
            prompt = "Human: 请写一个简短准确的含有关于{}和{}的隐喻（metaphor）的中文句子。\n\nAssistant: ".format(tenor,
                                                                                                                vehicle)

            trials = 0
            while metaphor == '' and trials < 5:
                trials += 1

                print('PROMPT: \n {}'.format(prompt))
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_new_tokens=100,
                                         do_sample=True, top_k=30, top_p=0.85,
                                         temperature=0.5, repetition_penalty=1.2,
                                         eos_token_id=2, bos_token_id=1, pad_token_id=0)

                response = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0].strip()

                response = response.replace(prompt, "")
                print('RESPONSE: \n {}'.format(response))
                metaphor = response.strip()

        results.append({'tenor': tenor,
                        'vehicle': vehicle,
                        'ground': ground,
                        'metaphor': metaphor})

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4, separators=(",", ": "))


def BELLE_Task2(CoT, sample_data, subset, output_file):
    results = []
    with open(os.path.join(data_path, 'train_data_sampled_200.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)[subset * 20:(subset + 1) * 20]

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "BELLE-7B-2M"))
    model = AutoModelForCausalLM.from_pretrained(os.path.join(model_path, "BELLE-7B-2M"))
    # model = model.eval()

    with open(os.path.join(data_path, sample_data), 'r', encoding='utf-8') as f:
        sampled_metaphors = json.load(f)

    vehicles = ['以{}作为本体（tenor）和以 {}作为共性（ground）构思隐喻时，'.format(m['tenor'], m['ground']) +
                '对应的喻体（vehicle）可以是：{}'.format(m['vehicle']) for m in sampled_metaphors]
    initial_prompt = 'Human: 在写作中，想构思隐喻（metaphor）时，有本体（tenor）和喻体（vehicle）的共性（ground）时就很容易构思对应的喻体，' + \
                     '诸如由下述本体和共性构思隐喻时，它们的喻体可以是以下这些：' + "，".join(vehicles)

    for entity in tqdm(data, total=len(data)):
        tenor = entity['tenor']
        vehicle = ''
        ground = entity['ground']
        metaphor = ''
        # print(tenor, vehicle)

        if CoT:
            prompt = "请参考我给你展示的例子的内容和格式，直接给出{}作为本体（tenor）和{}作为共性（ground）".format(tenor,
                                                                                                               ground) + \
                     "时的一个喻体（vehicle），不需要多加任何其他的解释或说明。\n\nAssistant: "

            trials = 0
            while vehicle == '' and trials < 5:
                trials += 1

                print('PROMPT: \n {}'.format(initial_prompt + "\n\n" + prompt))
                input_ids = tokenizer(initial_prompt + "\n\n" + prompt, return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_new_tokens=100,
                                         do_sample=True, top_k=30, top_p=0.85,
                                         temperature=0.5, repetition_penalty=1.2,
                                         eos_token_id=2, bos_token_id=1, pad_token_id=0)

                response = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0].strip()
                response = response.replace(initial_prompt + "\n\n" + prompt, "")

                print('RESPONSE: \n {}'.format(response))
                responses = re.split(r'[:：]', response)
                if len(responses) > 1:
                    for r in responses:
                        if '喻体' not in r:
                            vehicle = r.strip()
                            break
                else:
                    vehicle = response.strip()

            prompt = "参考共性{}，并基于你给出的喻体：“{}”，".format(ground, vehicle) + \
                     "写一个简短准确的含有关于{}和{}的隐喻（metaphor）的中文句子。\n\nAssistant: ".format(tenor, vehicle)

            trials = 0
            while metaphor == '' and trials < 5:
                trials += 1

                print('PROMPT: \n {}'.format(initial_prompt + "\n\n" + prompt))
                input_ids = tokenizer(initial_prompt + "\n\n" + prompt, return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_new_tokens=100,
                                         do_sample=True, top_k=30, top_p=0.85,
                                         temperature=0.5, repetition_penalty=1.2,
                                         eos_token_id=2, bos_token_id=1, pad_token_id=0)

                response = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0].strip()
                response = response.replace(initial_prompt + "\n\n" + prompt, "")

                print('RESPONSE: \n {}'.format(response))
                if '喻体' not in response and '隐喻' not in response:
                    metaphor = response.strip()
        else:
            prompt = "Human: 请写一个简短准确的含有关于{}并强调其{}的隐喻（metaphor）的中文句子。\n\nAssistant: ".format(
                tenor, ground)

            trials = 0
            while metaphor == '' and trials < 5:
                trials += 1

                print('PROMPT: \n {}'.format(prompt))
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_new_tokens=100,
                                         do_sample=True, top_k=30, top_p=0.85,
                                         temperature=0.5, repetition_penalty=1.2,
                                         eos_token_id=2, bos_token_id=1, pad_token_id=0)

                response = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0].strip()

                print('RESPONSE: \n {}'.format(response))
                response = response.replace(prompt, "")
                metaphor = response.strip()

        results.append({'tenor': tenor,
                        'vehicle': vehicle,
                        'ground': ground,
                        'metaphor': metaphor})

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4, separators=(",", ": "))


def combine_subsets(subset_paths, output_file):
    results = []
    for subset_path in subset_paths:
        with open(subset_path, 'r', encoding='utf-8') as f:
            subset = json.load(f)
        results += subset

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, separators=(",", ": "))


if __name__ == '__main__':

    for i in range(10):
        BELLE_Task1(True, 'train_clustered_sampled.json', i,
                      'results/belle_task1_CoT_non_conversation_results_{}.json'.format(i))
        gc.collect()
    combine_subsets(['results/belle_task1_CoT_non_conversation_results_{}.json'.format(i) for i in range(10)],
                     'results/belle_task1_CoT_non_conversation_results.json')

    for i in range(10):
        BELLE_Task1(True, 'train_clustered_sampled_embeddings.json', i,
                    'results/belle_task1_CoT_non_conversation_embeddings_results_{}.json'.format(i))
        gc.collect()
    combine_subsets(
        ['results/belle_task1_CoT_non_conversation_embeddings_results_{}.json'.format(i) for i in range(10)],
        'results/belle_task1_CoT_non_conversation_embeddings_results.json')

    for i in range(10):
        BELLE_Task1(False, 'train_clustered_sampled.json', i,
                      'results/belle_task1_non_conversation_results_{}.json'.format(i))
        gc.collect()
    combine_subsets(['results/belle_task1_non_conversation_results_{}.json'.format(i) for i in range(10)],
                     'results/belle_task1_non_conversation_results.json')

    for i in range(10):
        BELLE_Task2(True, 'train_clustered_sampled.json', i,
                    'results/belle_task2_CoT_non_conversation_results_{}.json'.format(i))
        gc.collect()
    combine_subsets(['results/belle_task2_CoT_non_conversation_results_{}.json'.format(i) for i in range(10)],
                    'results/belle_task2_CoT_non_conversation_results.json')

    for i in range(10):
        BELLE_Task2(True, 'train_clustered_sampled_embeddings.json', i,
                    'results/belle_task2_CoT_non_conversation_embeddings_results_{}.json'.format(i))
        gc.collect()
    combine_subsets(
        ['results/belle_task2_CoT_non_conversation_embeddings_results_{}.json'.format(i) for i in range(10)],
        'results/belle_task2_CoT_non_conversation_embeddings_results.json')

    for i in range(10):
        BELLE_Task2(False, 'train_clustered_sampled.json', i,
                    'results/belle_task2_non_conversation_results_{}.json'.format(i))
        gc.collect()
    combine_subsets(['results/belle_task2_non_conversation_results_{}.json'.format(i) for i in range(10)],
                    'results/belle_task2_non_conversation_results.json')