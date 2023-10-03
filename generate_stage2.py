import collections
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='6'
from sys import path
path.append(os.getcwd())
# path.append('/home/Medical_Understanding/MSL')
from typing import List
import time
import heapq
from tqdm import tqdm

import argparse
import glob
import logging
import math
import numpy as np
import torch
import transformers as tfs
from transformers import BertTokenizer, T5ForConditionalGeneration, Text2TextGenerationPipeline

from data_utils import data_collator, reader_dataset
from data_utils import utils as du
from data_utils import data_class
from data_utils import read_json, write_json
from utils import checkpoint
from utils import dist_utils
from utils import model_utils
from utils import options
from utils import sampler
from utils import utils
from preprocess_data import DatasetReader
from tensorboardX import SummaryWriter
from evaluate import evaluate_for_file

try:
    from apex import amp
except:
    pass

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class ModelTrainer(object):
    
    def __init__(self, args):

        utils.print_section_bar('Initializing components for training')

        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model_cfg)
        # tokenizer.add_special_tokens(
        #     {'additional_special_tokens': config.TOKENS})

        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['possible_statues']})

        cfg = tfs.T5Config.from_pretrained(args.pretrained_model_cfg)
        model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_cfg)
        if cfg.vocab_size != len(tokenizer):
            logger.info(f"Resize embedding from {cfg.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        model.load_state_dict(torch.load(args.model_recover_path, map_location=args.device))
        model.to(args.device)
        self.model = model


        optimizer = None

        self.start_epoch = 0
        self.start_offset = 0
        self.global_step = 0
        self.args = args
        
        self.tokenizer = tokenizer

    def get_eval_data_loader(self, eval_dataset):
        if torch.distributed.is_initialized():
            eval_sampler = sampler.SequentialDistributedSampler(
                eval_dataset,
                num_replicas=self.args.distributed_world_size,
                rank=self.args.local_rank)
        else:
            assert self.args.local_rank == -1
            eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.dev_batch_size,
            pin_memory=True,
            sampler=eval_sampler,
            num_workers=0,
            collate_fn=data_collator.collate_fn,
            drop_last=False)

        return dataloader

    def validate(self):
        args = self.args

        eval_dataset = reader_dataset.ReaderMedDataset_gen(args.dev_file, self.tokenizer, stage='stage2', stage1_index_file=args.stage1_index_file)
        # eval_dataset = reader_dataset.ReaderMedDataset_gen(args.dev_file, self.tokenizer, stage='stage2', stage1_index_file=None)
        eval_dataloader = self.get_eval_data_loader(eval_dataset)

        all_results = []
        # for step, batch in enumerate(tqdm(eval_dataloader)):
        for step, batch in enumerate(eval_dataloader):
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_masks, labels, dial_id,  window_id, term_id = batch
            if args.local_rank != -1:
                with self.model.no_sync(), torch.no_grad():
                    outputs = self.model.generate(inputs=input_ids,attention_mask=input_masks, max_length=64)
            else:
                with torch.no_grad():
                    outputs = self.model.generate(input_ids,attention_mask=input_masks, max_length=64, num_beams=1, num_return_sequences=1)
            # outputs = outputs.reshape(-1,4,64)
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # self.tokenizer.decode(labels[0])
            dial_id = list(dial_id.detach().cpu())
            window_id = list(window_id.detach().cpu())
            term_id = list(term_id.detach().cpu())
            all_results.extend(zip(generated_texts, dial_id, window_id, term_id))

        return all_results


def post_process(predicts, data_dir):
    post_predicts = []
    reader = DatasetReader(data_dir=data_dir)
    origin_values = list(reader.value_ids.keys())
    all_terms = list(reader.term_ids.keys())
    detail_states = reader.detail_states
    all_states = []
    for key, values in detail_states.items():
        for value in values:
            if value not in all_states and len(value) > 1:
                all_states.append(value)
    # all_states.append('没有提到')
    for predict in predicts:
        generated_text, dial_id, window_id, term_id = predict
        generated_text = ''.join(generated_text.split(' '))
        if generated_text[:2] in all_states:
            generated_text = generated_text[:2]
        elif generated_text[:4] in all_states:
            generated_text = generated_text[:4]
        elif generated_text[:5] in all_states:
            generated_text = generated_text[:5]
        else:
            continue

        dial_id = int(dial_id)
        window_id = int(window_id)
        term_id = int(term_id)
        post_predicts.append([dial_id, window_id, term_id, generated_text])
    return post_predicts, reader



def main():
    parser = argparse.ArgumentParser()

    options.add_model_params(parser)
    options.add_cuda_params(parser)
    options.add_training_params(parser)
    options.add_data_params(parser)
    args = parser.parse_args()
    args.data_dir = os.path.join(args.origin_data_dir, args.data_dir)
    # try:
    #     args.local_rank = int(os.environ["LOCAL_RANK"])
    # except:
    #     args.local_rank = -1
    if args.add_category:
        args.data_dir += '_add_category'
        args.output_dir += '_add_category'
    if args.add_state:
        args.data_dir += '_add_state'
        args.output_dir += '_add_state'
    args.model_recover_dir = os.path.join(args.output_dir, args.model_recover_dir)
    args.log_dir = os.path.join(args.output_dir, args.log_dir)
    
    args.train_file = os.path.join(args.data_dir, args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)


    assert os.path.exists(args.pretrained_model_cfg), \
        (f'{args.pretrained_model_cfg} doesn\'t exist. '
         f'Please manually download the HuggingFace model.')
    options.setup_args_gpu(args)
    # Makes sure random seed is fixed.
    # set_seed must be called after setup_args_gpu.
    options.set_seed(args)

    if dist_utils.is_local_master():
        utils.print_args(args)
    mode = args.dev_file.split('/')[-1].split('.')[0]
    model_dir = args.model_recover_dir.split('/model')[0]
    eval_results = []
    for i in range(0, 100):
        args.model_recover_path = args.model_recover_dir.format(i)
        args.stage1_index_file = os.path.join(model_dir, 'pred_stage1_{}_{}.json'.format(mode, i))
        trainer = ModelTrainer(args)
        predicts = trainer.validate()
        post_predicts, reader = post_process(predicts, data_dir=args.origin_data_dir)

        ontology = reader.ontology
        origin_values = list(reader.value_ids.keys())
        detail_values = reader.detail_states
        all_terms = []
        term2category = {}
        for category, terms in ontology.items():
            if category == '状态':
                continue
            all_terms.extend(terms)
            for term in terms:
                term2category[term] = category

        dialogues = read_json(path=os.path.join(args.origin_data_dir, '{}.json'.format(mode)))
        new_dialogues = []
        for dialogue in dialogues:
            new_dialogue = []
            for window in dialogue:
                window['pred'] = []
                new_dialogue.append(window)
            new_dialogues.append(new_dialogue)
        
        for predict in post_predicts:
            dial_id, window_id, term_id, generated_text = predict
            term = all_terms[term_id]
            category = term2category[term]
            try:
                value = origin_values[detail_values[category].index(generated_text)]
            except:
                continue
            generated_text = '{}:{}-状态:{}'.format(category, term, value)
            new_dialogues[dial_id][window_id]['pred'].append(generated_text)
        
        write_json(data=new_dialogues, path=os.path.join(model_dir, 'pred_stage_all_{}_{}_two_stages.json'.format(mode, i)))
        eval_result = evaluate_for_file(eval_file=os.path.join(model_dir, 'pred_stage_all_{}_{}_two_stages.json'.format(mode, i)))
        eval_result['epoch'] = i
        eval_results.append(eval_result)
    
        write_json(data=eval_results, path=os.path.join(args.output_dir, '{}_eval_results_two_stages.json'.format(mode)))


if __name__ == '__main__':
    main()
