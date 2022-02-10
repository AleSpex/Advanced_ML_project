import json

file_list = ["Cartoon_descriptions.json", "Photo_descriptions.json", "Sketch_descriptions.json"]
category_list = ["cartoon", "photo", "sketch"]
new_labels = open("new_labels.json", "a")
new_labels.write("[")
domain_vect = ["house", "guitar", "horse", "dog", "elephant", "giraffe", "person"]
for i in range(len(file_list)):
    with open(file_list[i]) as f:
        data = json.load(f)
        counter = 0
        while counter < 7:
            for pic in data[domain_vect[counter]]:
                data_dict = {}
                data_dict['image_name'] = f"{category_list[i]}/{domain_vect[counter]}/"+pic
                data_dict['category'] = category_list[i]
                data_dict['descriptions'] = []
                for values in data[domain_vect[counter]][pic].values():
                    data_dict['descriptions'].append(values)
                dict_string = json.dumps(data_dict)
                new_labels.write(dict_string)
                new_labels.write(",")
            counter += 1

new_labels.truncate(new_labels.tell() - 1)
        
#new_labels.write("]")
new_labels.close()