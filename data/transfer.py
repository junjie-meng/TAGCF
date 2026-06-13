import numpy as np
import pickle
from scipy.sparse import coo_matrix, csr_matrix
import json
from collections import Counter

dataset = 'yelp'

attr_path = f'/home1/workspace/mengjunjie/lightgcn/data/{dataset}/preference.txt'
id2user_path = f'/home1/workspace/mengjunjie/lightgcn/data/{dataset}/id2user.json'
id2item_path = f'/home1/workspace/mengjunjie/lightgcn/data/{dataset}/id2item.json'
output_path = f'{dataset}/attr_edges.pkl'
attr_edges = []
attr_counter = Counter()
with open(attr_path, 'r') as f:
    for line in f:
        item, attr = line.split(' ')
        attr_edges.append((int(item), int(attr)))
        attr_counter[int(attr)] += 1

print(len(attr_edges))
## transfer to coo_matrix 
item_ids, attr_ids = zip(*attr_edges)

item_ids = np.array(item_ids)
attr_ids = np.array(attr_ids)
with open(id2item_path, 'r') as f1,\
     open(id2user_path, 'r') as f2 :
    id2item = json.load(f1)
    id2user = json.load(f2)
user_item_num = len(id2user) + len(id2item)

print(user_item_num)
print(max(attr_ids)+1)

attr_edges = coo_matrix((np.ones(len(item_ids)), (item_ids, attr_ids)), shape=(user_item_num, max(attr_ids) + 1))
with open(output_path, 'wb') as f:
    pickle.dump(attr_edges, f)