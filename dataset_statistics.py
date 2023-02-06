import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn3


def open_files(filename):
    f = open(filename + '.json')
    filename = json.load(f)
    return filename


def attributes_for_parent_type(parent2type, type2adj):
    parent2attrib = defaultdict(list)
    for par, typ in parent2type.items():
        for t in typ:
            try:
                parent2attrib[par] += type2adj[t]
            except KeyError:
                pass

    return parent2attrib


def attribute_type_count(dataset, parent2attrib):
    attrib_count = {'pos': defaultdict(int), 'neg': defaultdict(int)}

    for i in dataset:
        for pos in i['positive_attributes']:
            try:
                attrib_count['pos'][[k for k, v in parent2attrib.items() if pos in v][0]] += 1
            except IndexError:
                attrib_count['neg']['other'] += 1

        for neg in i['negative_attributes']:
            try:
                attrib_count['neg'][[k for k, v in parent2attrib.items() if neg in v][0]] += 1
            except IndexError:
                attrib_count['neg']['other'] += 1

    return attrib_count


def attribute_type_freq(dataset, parent2type, type2adj):

    parent2attrib = attributes_for_parent_type(parent2type, type2adj)
    attr_count = attribute_type_count(dataset, parent2attrib)
    tot = len(dataset)

    for key, val in attr_count.items():
        for att_type in val.keys():
            attr_count[key][att_type] /= tot

    return attr_count


def attribute_sign_freq(dataset):
    pos = 0
    neg = 0
    tot = len(dataset)

    for i in dataset:
        pos += len(i['positive_attributes'])
        neg += len(i['negative_attributes'])

    return pos / tot, neg / tot


def object_dist(dataset):
    obj_list = []
    objects = defaultdict(int)

    for i in dataset:
        if i['object_name'] not in obj_list:
            obj_list.append(i['object_name'])
        if i['object_name'] not in objects:
            objects[i['object_name']] = 1
        else:
            objects[i['object_name']] += 1

    return obj_list, objects


def plot_attr_freq(train, test, val, attr='pos'):
    # attr can be 'pos' or 'neg'

    mydicts = [train[attr], test[attr], val['pos']]
    df = pd.concat([pd.Series(d) for d in mydicts], axis=1).fillna(0).T
    df.index = ['Train', 'Test', 'Val']
    df1 = df.T
    df1.sort_index(inplace=True)
    df1.plot.bar(rot=15, title="{} attribute distribution per Dataset".format(attr), colormap='Paired')
    plt.savefig("{}DD.png".format(attr))

    plt.show(block=True)


def venn_diagram(train, test, val):
    plt.figure(figsize=(10, 10))
    plt.title("Venn Diagram of object_names")

    venn3(subsets=[set(train), set(test), set(val)], set_labels=('Train', 'Test', 'Validation'))
    plt.savefig("Venn_objects.png")
    plt.show()


if __name__ == "__main__":

    file_list = ['attribute_index', 'attribute_parent_types', 'attribute_types', 'test_data', 'train_data', 'val_data']

    for i in file_list:
        exec("%s = %s" % (i, open_files(i)))

    # Get the total dataset (#10595)
    whole_data = train_data + val_data + test_data

    # Get the attribute frequency distribution
    attr_freq = attribute_type_freq(whole_data, attribute_parent_types, attribute_types)
    attr_freq_train = attribute_type_freq(train_data, attribute_parent_types, attribute_types)
    attr_freq_test = attribute_type_freq(test_data, attribute_parent_types, attribute_types)
    attr_freq_val = attribute_type_freq(val_data, attribute_parent_types, attribute_types)

    plot_attr_freq(attr_freq_train, attr_freq_test, attr_freq_val, attr='pos')
    plot_attr_freq(attr_freq_train, attr_freq_test, attr_freq_val, attr='neg')

    # Get average number of attributes per instance
    pos_freq, neg_freq = attribute_sign_freq(whole_data)
    pos_freq_train, neg_freq_train = attribute_sign_freq(train_data)
    pos_freq_test, neg_freq_test = attribute_sign_freq(test_data)
    pos_freq_val, neg_freq_val = attribute_sign_freq(val_data)

    # Get object class distribution
    whole_obj, object_whole_dict = object_dist(whole_data)
    train_obj, object_train_dict = object_dist(train_data)
    test_obj, object_test_dict = object_dist(test_data)
    val_obj, object_val_dict = object_dist(val_data)

    venn_diagram(train_obj, test_obj, val_obj)
