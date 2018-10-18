import json


with open("dataset_slo.json", "r") as read_file:
    data = json.load(read_file)

with open('data.txt', 'w') as f:
    f.write(data)