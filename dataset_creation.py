# Dataset taken from https://github.com/adobe-research/vaw_dataset

import json
import os
import random


def current_path():
    print("Current working directory:")
    print(os.getcwd())
    print()


def open_files(filename):
    f = open(filename + '.json')
    filename = json.load(f)
    return filename


def get_objects(data):
    objects = {}
    for i in data:
        if i['object_name'] not in objects:
            objects[i['object_name']] = 1
        else:
            objects[i['object_name']] += 1
    return objects


def get_test_general(objects_num):
    i = 0
    obj = []
    while i < objects_num:
        name = random.sample(list(less_10_dict), 1)[0]
        partial = objects_num - i
        if less_10_dict[name] <= partial and name not in obj:
            i += less_10_dict[name]
            obj.append(name)
    return obj


def get_data(count, instance_id, test=None, val=None):
    ele_num = 0
    dataset = []
    if test:
        dataset += test

    while ele_num < count:
        ele = random.sample(instance_id, 1)[0]
        if ele not in dataset and ele not in val and ele not in test:
            dataset.append(ele)
            ele_num += 1

    return dataset


if __name__ == "__main__":
    random.seed(42)
    current_path()

    file_list = ['attribute_index', 'attribute_parent_types', 'attribute_types', 'test',
                 'train_part1', 'train_part2', 'val']

    for i in file_list:
        exec("%s = %s" % (i, open_files(i)))

    # Get the total number of instances (# 260895)
    whole_data = train_part1 + train_part2 + val + test
    instance_id = []
    for i in whole_data:
        instance_id.append(i['instance_id'])

    # Create a dictionary of all the objects in the VAW Dataset and sort them by frequency
    objects = get_objects(whole_data)
    sortedDict = sorted(objects)
    less_10_dict = {k: v for k, v in objects.items() if v < 10}

    ### CREATION OF THE NEW DATASET
    # The new dataset respect the proportions of the original dataset
    # Training set = 8800 instances
    # Test set with unseen objects for generalization (30%) = 1295 instances
    # Validation set = 500 instances

    train_key_count = 8800
    val_key_count = 500
    test_key_count = 1295 - 380

    # Select the objects to represent the 30% of the test set
    generalization_set = get_test_general(380)
    generalization_id = []

    for i in whole_data:
        if i['object_name'] in generalization_set:
            generalization_id.append(i['instance_id'])

    # Random Sample Training, Validation and Test Data
    test_keys = get_data(test_key_count, instance_id, test=generalization_id)
    val_keys = get_data(val_key_count, instance_id, test=test_keys)
    train_keys = get_data(train_key_count, instance_id, test=test_keys, val=val_keys)

    # Creation of .json files
    test_final_json = [i for i in whole_data if i['instance_id'] in test_keys]
    val_final_json = [i for i in whole_data if i['instance_id'] in val_keys]
    train_final_json = [i for i in whole_data if i['instance_id'] in train_keys]

    with open('test_data.json', 'w') as outfile:
        json.dump(test_final_json, outfile)

    with open('val_data.json', 'w') as outfile:
        json.dump(val_final_json, outfile)

    with open('train_data.json', 'w') as outfile:
        json.dump(train_final_json, outfile)
