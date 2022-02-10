import json

output = '{"test":[],"val":[],"train":['

with open("image_descriptions.json", "r") as f:
    data = json.load(f)
    for item in data:
        name = item['image_name']
        output += f'"{name}",'
    output += ']}'

with open("output.json", "w") as fout:
    fout.write(output)