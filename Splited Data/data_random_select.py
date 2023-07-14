import json
import random


def sample_n_from_dataset(file_path, output_path, n, seed=42):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(seed)
    sampled_data = random.sample(data, n)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4, separators=(",", ": "))


def sample_x_percent_from_dataset(file_path, output_path, x, seed=42):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(seed)
    num_samples = int(len(data) * x)
    sampled_data = random.sample(data, num_samples)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4, separators=(",", ": "))


if __name__ == '__main__':
    n = 20
    input_file_path = 'train_data.json'
    output_file_path = 'train_data_sampled_%d.json' % n
    # input_file_path = 'val_data.json'
    # output_file_path = 'val_data_sampled_%d.json' % n
    sample_n_from_dataset(input_file_path, output_file_path, n)
    # sample_x_percent_from_dataset('your_file_path.json', )
