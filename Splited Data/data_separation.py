import json

# 加载数据
with open('annotated_corpora_separated.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


print('数据大小:',len(data))
# 计算划分点
train_split = int(len(data) * 0.8)
val_split = int(len(data) * 0.9)

# 划分数据
train_data = data[:train_split]
val_data = data[train_split:val_split]
test_data = data[val_split:]

print('train_data数据大小:',len(train_data))
print('val_data数据大小:',len(val_data))
print('test_data数据大小:',len(test_data))

# 将划分后的数据保存为新的json文件
with open('train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4, separators=(",", ": "))

with open('val_data.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4, separators=(",", ": "))

with open('test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4, separators=(",", ": "))

