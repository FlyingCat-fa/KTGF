# -*- coding: utf-8 -*-
"""
DataLoder For Chinese Medical Dialogue Dataset
将CMDD数据集转换为MIE格式
"""
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='3'
from sys import path
path.append(os.getcwd())
path.append('/home/Framework_hu/dataset/CMDD_v0')

def getdata():
    train_document_number=1241
    test_document_number=413
    dev_document_number=413
    traindata={}
    testdata={}
    devdata={}
    for i in range(train_document_number):
        with open("train/train_"+str(i)+'.json',encoding="utf8") as f:
            traindocument=json.load(f)
            traindata[traindocument['example-id']]=traindocument['dialogue-content']
    for i in range(test_document_number):
        with open("test/test_"+str(i)+'.json',encoding="utf8") as f:
            testdocument=json.load(f)
            testdata[testdocument['example-id']]=testdocument['dialogue-content']
    for i in range(dev_document_number):
        with open("dev/dev_"+str(i)+'.json',encoding="utf8") as f:
            devdocument=json.load(f)
            devdata[devdocument['example-id']]=devdocument['dialogue-content']
    return traindata,testdata,devdata

def find_description(example):
    sentence = example['sentence']
    label = example['label']
    entity = example['normalized']

    sentence = list(sentence)
    label = list(label)
    description = []
    state = 'O'
    for i, string in enumerate(label):
        if string == 'B':
            state = 'B'
            description.append(sentence[i])
        elif string == 'I':
            if state == 'B':
                description[-1] += sentence[i]
            elif state == 'O':
                description.append(sentence[i])
            else:
                print('数据异常！！！')
        else:
            state = 'O'
    
    if len(description) > len(entity):
        description = description[:len(entity)]

    return description

def entity_normal(entity):
    if entity == "发":
        entity = "发热"
    if entity == '出汉':
        entity = "出汗"
    if entity == "幼儿急诊":
        entity = "幼儿急疹"
    if entity == "呼吸深快":
        entity = "呼吸急促"
    if entity == "便隐血":
        entity = "大便隐血"
    if entity == "稀水便":
        entity = "稀便"
    if entity == "便隐血":
        entity = "大便隐血"
    if entity == "腹胀症":
        entity = "腹胀"
    if entity == "胃肠炎":
        entity = "肠胃炎"
    if entity == "先天性心脏病":
        entity = "心脏病"
    if entity == "支气管炎肺炎":
        entity = "支气管肺炎"
    if entity == "鼻出血":
        entity = "鼻流血"
    # if entity == "鼻出血":
    #     entity = "鼻流血"
    # if entity in ["2", "发"]:
    #     print('{} 存在异常实体'.format(key))
    return entity

def state_normal(labels):
    for label in labels:
        str1, str2 = label.split('-')
        entity = str1.split(':')
        entity = str1.split(':')


def dialog2MIE(dialog):
    state_dict= {"1":'阳性', "2":'阴性', "3":'未知'}
    dialog_MIE = []
    for i, sen in enumerate(dialog):
        window = {}
        window['utterances'] = []
        window['label'] = []
        labels = {}
        for j in [4,3,2,1,0]:
            if i-j < 0:
                window['utterances'].append('')
            else:
                sentence = dialog[i-j]
                utterance = '{}:{}'.format(sentence['speaker'], sentence['sentence'])
                window['utterances'].append(utterance)

                for j, entity in enumerate(sentence["normalized"]):
                    if entity == "2":
                        continue
                    entity = entity_normal(entity)
                    try:
                        labels[entity] = state_dict[sentence['type'][j]]
                    except:
                        print("状态不匹配！！")
                        labels[entity] = state_dict["1"]
        
        for key, value in labels.items():
            label = "症状:{}-状态:{}".format(key, value)
            window['label'].append(label)
        
        dialog_MIE.append(window)

    return dialog_MIE


def CMDD2MIE(dataset, mode='train'):
    data = []
    for key in dataset.keys():
        dialog = dataset[key]
        dialog_MIE = dialog2MIE(dialog)
        data.append(dialog_MIE)
    
    with open('{}.json'.format(mode), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print('已写入{}文件 !'.format(mode))


            



traindata,testdata,devdata=getdata()
totaldata={}
totaldata.update(traindata)
totaldata.update(testdata)
totaldata.update(devdata)

ontology = {}
ontology["症状"] = []
example_dict = {}
example_dict["症状"] = {}
for key in totaldata.keys():
    examples = totaldata[key]
    
    for i, sentence in enumerate(examples):
        description = find_description(sentence)
        for j, entity in enumerate(sentence["normalized"]):
            if entity == "2":
                continue
            entity = entity_normal(entity)
            if entity not in ontology["症状"]:
                ontology["症状"].append(entity)
                example_dict["症状"][entity] = 1
            else:
                example_dict["症状"][entity] += 1
            # try:
            #     if description[j] not in example_dict["症状"][entity]:
            #         example_dict["症状"][entity].append(description[j])
            # except:
            #     print('description 异常！')
# terms = []
# for key, value in example_dict["症状"].items():
#     terms.append((key, value))

by_value = sorted(example_dict["症状"].items(),key = lambda item:item[1])

ontology["状态"] = ["阳性", "阴性", "未知"]

with open('ontology.json', 'w', encoding='utf-8') as f:
    json.dump(ontology, f, indent=2, ensure_ascii=False)

with open('example_dict_all.json', 'w', encoding='utf-8') as f:
    json.dump(example_dict, f, indent=2, ensure_ascii=False)

CMDD2MIE(traindata, mode='train')
CMDD2MIE(devdata, mode='dev')
CMDD2MIE(testdata, mode='test')






