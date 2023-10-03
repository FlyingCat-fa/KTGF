import collections
import json
import os
from sys import path
path.append(os.getcwd())
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
from transformers import BertTokenizer, T5ForConditionalGeneration

from data_utils import data_collator, reader_dataset
from utils import dist_utils
from utils import model_utils
from utils import options
from utils import sampler
from utils import utils
from tensorboardX import SummaryWriter

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
        # one stage
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<eot>']})

        cfg = tfs.BertConfig.from_pretrained(args.pretrained_model_cfg)
        model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_cfg)
        if cfg.vocab_size != len(tokenizer):
            logger.info(f"Resize embedding from {cfg.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        # model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.70.bin'), map_location=args.device))
        if args.inference_only:
            optimizer = None
        else:
            optimizer = model_utils.get_optimizer(
                model,
                learning_rate=args.learning_rate,
                adam_eps=args.adam_eps,
                weight_decay=args.weight_decay)

        self.model, self.optimizer = model_utils.setup_for_distributed_mode(
            model,
            optimizer,
            args.device,
            args.n_gpu,
            args.local_rank,
            args.fp16,
            args.fp16_opt_level)

        self.start_epoch = 0
        self.start_offset = 0
        self.global_step = 0
        self.args = args
        
        self.tokenizer = tokenizer
        self.tb_writer = SummaryWriter(logdir=args.log_dir)


    def get_train_dataloader(self, train_dataset, shuffle=True, offset=0):
        if torch.distributed.is_initialized():
            train_sampler = sampler.DistributedSampler(
                train_dataset,
                num_replicas=self.args.distributed_world_size,
                rank=self.args.local_rank,
                shuffle=shuffle)
            train_sampler.set_offset(offset)
        else:
            assert self.args.local_rank == -1
            train_sampler = torch.utils.data.RandomSampler(train_dataset)

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=data_collator.collate_fn,
            drop_last=False)

        return dataloader

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

    def run_train(self):
        args = self.args

        train_dataset = reader_dataset.ReaderMedDataset(args.train_file, self.tokenizer)
        train_dataloader = self.get_train_dataloader(
            train_dataset,
            shuffle=True,
            offset=self.start_offset)

        updates_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps)
        total_updates = updates_per_epoch * args.num_train_epochs

        dataloader_steps = self.start_offset // (
            args.distributed_world_size * args.batch_size)
        updated_steps = (dataloader_steps // 
                         args.gradient_accumulation_steps) + (
                             self.start_epoch * updates_per_epoch)
        remaining_updates = total_updates - updated_steps

        # global_step is added per dataloader step.
        calc_global_step = (self.start_epoch * len(train_dataloader) + 
                            dataloader_steps)

        assert self.global_step == calc_global_step, \
            (f'global step = {self.global_step}, '
             f'calc global step = {calc_global_step}')

        self.scheduler = model_utils.get_schedule_linear(
            self.optimizer,
            warmup_steps=args.warmup_steps,
            training_steps=total_updates,
            last_epoch=self.global_step-1)

        utils.print_section_bar('Training')
        if dist_utils.is_local_master():
            logger.info(f'Total updates = {total_updates}')
            logger.info(
                f'Updates per epoch (/gradient accumulation) = '
                f'{updates_per_epoch}')
            logger.info(
                f'Steps per epoch (dataloader) = {len(train_dataloader)}')
            logger.info(
                f'Gradient accumulation steps = '
                f'{args.gradient_accumulation_steps}')
            logger.info(
                f'Start offset of the epoch {self.start_epoch} (dataset) = '
                f'step {self.start_offset}')
            logger.info(
                f'Updated step of the epoch {self.start_epoch} (dataloader) = '
                f'step {updated_steps}')
            logger.info(
                f'Total remaining updates = {remaining_updates}')

        # Starts training here.
        for epoch in range(self.start_epoch, int(args.num_train_epochs)):
            utils.print_section_bar(f'Epoch {epoch}')

            if isinstance(train_dataloader.sampler, sampler.DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            self._train_epoch(epoch, train_dataloader)

            if isinstance(train_dataloader.sampler, sampler.DistributedSampler):
                train_dataloader.sampler.set_offset(0)

        utils.print_section_bar('Training finished.')

        return


    def _train_epoch(self, epoch, train_dataloader):
        args = self.args
        epoch_loss = 0
        rolling_train_losses = collections.defaultdict(int)
        rolling_train_others = collections.defaultdict(int)

        step_offset = 0

        train_batch_times = []
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # for step, batch in enumerate(epoch_iterator, start=step_offset):
        for step, batch in enumerate(train_dataloader, start=step_offset):
            self.model.train()
            step += 1

            batch_start_time = time.time()
            if step % args.gradient_accumulation_steps != 0 \
                    and args.local_rank != -1:
                with self.model.no_sync(): # 不进行梯度同步
                    losses, others = self._training_step(batch)
            else:
                losses, others = self._training_step(batch)
            batch_end_time = time.time()
            train_batch_times.append(batch_end_time - batch_start_time)

            self.global_step += 1

            '''
                record loss
            '''
            epoch_loss += losses['total']
            for k, loss in losses.items():
                # add
                if dist_utils.is_local_master():
                    self.tb_writer.add_scalar(k + '_loss', loss, self.global_step)

                rolling_train_losses[k] += loss
            for k, other in others.items():
                # other could be -1 if adv_loss not applicable
                rolling_train_others[k] += max(other, 0)

            '''
                parameters update
            '''
            if (step - step_offset) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), args.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), args.max_grad_norm
                        )
    
                self.scheduler.step()
                self.optimizer.step()
                self.model.zero_grad()

        epoch_loss = epoch_loss / len(train_dataloader)

        if dist_utils.is_local_master():
            logger.info(f'Train: global step = {self.global_step}; '
                        f'step = {step}')
            logger.info(f'Avg. total Loss of epoch {epoch} ={epoch_loss:.3f}')
            logger.info(
                "** ** * Saving fine-tuned model ** ** * ")
            output_model_file = os.path.join(
                args.output_dir, "model.{0}.bin".format(epoch))
            if hasattr(self.model, 'module'):
                torch.save(self.model.module.state_dict(), output_model_file)
            else:
                torch.save(self.model.state_dict(), output_model_file)

    def _training_step(self, batch) -> torch.Tensor:
        args = self.args
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_masks, labels, dial_id,  window_id, term_id = batch
        output = self.model(input_ids=input_ids, attention_mask=input_masks, labels=labels)
        losses = {'total': output['loss']}

        losses = {k: loss.mean() for k, loss in losses.items()}

        if args.fp16:
            with amp.scale_loss(losses['total'], self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses['total'].backward()

        return {k: v.item() for k, v in losses.items()}, {}


def main():
    parser = argparse.ArgumentParser()

    options.add_model_params(parser)
    options.add_cuda_params(parser)
    options.add_training_params(parser)
    options.add_data_params(parser)
    args = parser.parse_args()

    if args.add_category:
        args.data_dir += '_add_category'
        args.output_dir += '_add_category'
    if args.add_state:
        args.data_dir += '_add_state'
        args.output_dir += '_add_state'
    args.log_dir = os.path.join(args.output_dir, args.log_dir)
    
    args.train_file = os.path.join(args.data_dir, args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    # try:
    #     args.local_rank = int(os.environ["LOCAL_RANK"])
    # except:
    #     args.local_rank = -1

    assert os.path.exists(args.pretrained_model_cfg), \
        (f'{args.pretrained_model_cfg} doesn\'t exist. '
         f'Please manually download the HuggingFace model.')
    options.setup_args_gpu(args)
    # Makes sure random seed is fixed.
    # set_seed must be called after setup_args_gpu.
    options.set_seed(args)

    if dist_utils.is_local_master():
        utils.print_args(args)

    trainer = ModelTrainer(args)

    if args.train_file is not None:
        trainer.run_train()
    else:
        logger.warning(
            'Neither train_file or (checkpoint_file & dev_file) parameters '
            'are specified. Nothing to do.')


if __name__ == '__main__':
    main()
