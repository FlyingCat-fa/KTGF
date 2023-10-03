from cgitb import text
from doctest import Example
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import json
import os
from collections import namedtuple
import random
import argparse

from collections import defaultdict, Counter
from data_utils.common_utils import read_json, write_json

import numpy as np
from tqdm import tqdm

import csv
import sys


DEFAULT_TYPE_ID = 0
USER_TYPE_ID = 1
AGENT_TYPE_ID = 2
TYPE_OFFSET = 2


class DatasetReader:

    def __init__(self,
                 data_dir: str = None,) -> None:
        self._datapath = data_dir
        self.ontology, self.term_ids, self.value_ids, self.detail_states, self.example_dict = self._load_ontology()

    def _load_ontology(self):
        ontology = read_json(os.path.join(self._datapath, 'ontology.json'))
        all_terms = []
        term_ids = {}
        all_values = []
        value_ids = {}
        for category, terms in ontology.items():
            if category == "状态":
                all_values.extend(terms)
                continue
            all_terms.extend(terms)
        for i, term in enumerate(all_terms):
            term_ids[term] = i
        for i, value in enumerate(all_values):
            value_ids[value] = i
        example_dict = read_json(os.path.join(self._datapath, 'example_dict.json'))
        if 'Chunyu' in self._datapath:
            detail_states = {'症状':['阳性', '阴性', '医生诊断有', '医生诊断无','未知'], \
                    '检查':['患者已做', '患者未做', '医生建议', '医生不建议', '未知'], \
                    '手术': ['患者已做', '患者未做', '医生建议', '医生不建议', '未知'], \
                    '一般信息': ['正常', '异常', '', '', '未知']}
        elif 'CMDD' in self._datapath:
            detail_states = {'症状':['阳性', '阴性','未知']}
        return ontology, term_ids, value_ids, detail_states, example_dict
    
    def sub_label_decode(self, sub_label):
        # "症状:心肌缺血-状态:未知"
        new_sub_label = sub_label.split('-')
        category, term = new_sub_label[0].split(':')
        value = new_sub_label[1].split(':')[1]
        return category, term, value

    def label_convert(self, label):
        new_label = {}
        for sub_label in label:
            category, term, value = self.sub_label_decode(sub_label)

            new_label['{}:{}'.format(category, term)] = value
        return new_label

    def window_context(self, window, max_len=448):
        utterances = window['utterances']
        context = ''
        for utterance in utterances:
            utterance = utterance.strip()[:200]
            if len(utterance) > 2 and utterance[-1] != '？':
                utterance += '。'
            context += utterance
        return context[:max_len]
    
    def term_sequence_convert(self, label, context, dial_id, window_id):
        terms = []
        for sub_label in label:
            category, term, value = self.sub_label_decode(sub_label)
            terms.append(term)
        example = {'context': context + '提到的医疗词汇', 'output': '，'.join(terms), 'dial_id': dial_id, 'window_id': window_id, 'term_id': -1}
        return example

    def stage2_convert(self, label, context, dial_id, window_id, add_state=False, add_category=False):
        examples = []
        for sub_label in label:
            category, term, value = self.sub_label_decode(sub_label)
            if value == '没有提到':
                continue
            if value != '没有提到':
                value = self.detail_states[category][self.value_ids[value]]

            if category == '一般信息':
                possible_value = self.detail_states[category][:2] + [self.detail_states[category][-1]]
            else:
                possible_value = self.detail_states[category]
            # if rectify:
            #     possible_value = possible_value + ['没有提到']
            possible_value = '，'.join(possible_value)
            prompt = term
            if add_category:
                prompt = category + ':' + prompt

            prompt += '的状态'
            if add_state:
                prompt = '候选状态包括{}。'.format(possible_value) + prompt
            # 症状腹痛如口语的状态。候选包括阳性，阴性，...
            example = {'context': context + prompt, 'output': value, 'dial_id': dial_id, 'window_id': window_id, 'term_id': self.term_ids[term]}
            examples.append(example)
        return examples
    

    def _load_conversations(self, mode: str, args):
        dialogues = read_json(os.path.join(self._datapath, '{}.json'.format(mode)))
        examples = []
        for dial_id, dialogue in enumerate(dialogues):
            for window_id, window in enumerate(dialogue):
                context = self.window_context(window)
                examples.append(self.term_sequence_convert(window['label'], context, dial_id, window_id))
                examples.extend(self.stage2_convert(window['label'] + window["pre_term"], context, dial_id, window_id, \
                                add_category=args.add_category, add_state=args.add_state))
        return examples


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    reader = DatasetReader(args.data_dir)
    for split in ["train", "dev", "test"]:
        examples = reader._load_conversations(mode=split, args=args)
        # write_json(data=examples[:100], path=os.path.join(args.out_dir, '{}_100.json'.format(split)))
        write_json(data=examples, path=os.path.join(args.out_dir, '{}.json'.format(split)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default='/home/Framework_hu/dataset/Chunyu',
        # default='/home/Framework_hu/dataset/CMDD',
        help="directory path of the data",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='/home/Framework_hu/dataset/Chunyu/processed',
        # default='/home/Framework_hu/dataset/CMDD/processed',
        help="directory path of the output data",
    )
    parser.add_argument(
        "--add_category",
        action='store_true',
        default=False,
        help="directory path of the output data",
    )
    parser.add_argument(
        "--add_state",
        action='store_true',
        default=False,
        help="directory path of the output data",
    )
    # parser.add_argument(
    #     "--rectify",
    #     action='store_true',
    #     default=False,
    #     help="directory path of the output data",
    # )
    args = parser.parse_args()
    # if args.add_category:
    #     args.out_dir += '_add_category'
    # if args.add_state:
    #     args.out_dir += '_add_state'
    # if args.rectify:
    #     args.out_dir += '_rectify'


    main(args)
