import json

# 读取原始数据
with open('annotated_corpora.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = []
error_items = []  # 用于存储出错的元素
for item in data:
    try:
        tenors = item['tenor'].split('；')
        vehicles = item['vehicle'].split('；')
        grounds = item['ground'].split('；')
        for i in range(len(tenors)):
            tenor = tenors[i]
            vehicle_list = vehicles[i].split('，')
            ground_list = grounds[i].split('，')
            if len(vehicle_list) != len(ground_list):
                raise ValueError("Mismatch between vehicle and ground counts")
            for j in range(len(vehicle_list)):
                vehicle = vehicle_list[j]
                ground = ground_list[j]
                # ground_sub_list = ground_list[j].split('、')
                new_item = item.copy()
                new_item['tenor'] = tenor
                new_item['vehicle'] = vehicle
                new_item['ground'] = ground
                new_item['sub_id'] = j
                new_data.append(new_item)

    except (IndexError, ValueError) as e:
        print(f"Error processing item: {item}")
        print(f"Error message: {str(e)}")
        error_items.append(item)  # 将出错的元素添加到列表中

# 保存处理后的数据
with open('annotated_corpora_separated.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

# 打印出所有出错的元素
print("Error items:")
for item in error_items:
    print(item)
