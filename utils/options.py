import argparse
import logging
from operator import truediv
import os
import random
import socket

import numpy as np
import torch

from models import loss
from utils import dist_utils

logger = logging.getLogger()


def add_data_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--do_lower_case',
        default=True,
        type=bool,
        help=('Whether to lower case the input text. True for uncased models, '
              'False for cased models.'))

def add_model_params(parser: argparse.ArgumentParser):
    """Common parameters to initialize an encoder-based model."""
    
    parser.add_argument(
        '--pretrained_model_cfg',
        default='pretrained_models/t5-small-chinese-cluecorpussmall',
        type=str,
        help='Path of the pre-trained model.')
    # parser.add_argument(
    #     '--pretrained_model_cfg',
    #     default='pretrained_models/t5-base-chinese-cluecorpussmall',
    #     type=str,
    #     help='Path of the pre-trained model.')
    parser.add_argument(
        '--checkpoint_file',
        default=None,
        type=str,
        help='Trained checkpoint file to initialize the model.')

    # parser.add_argument(
    #     '--checkpoint_file',
    #     default='Les/exp/best_em',
    #     type=str,
    #     help='Trained checkpoint file to initialize the model.')
    parser.add_argument(
        '--projection_dim',
        default=0,
        type=int,
        help='Extra linear layer on top of standard bert/roberta encoder.')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=428,
        help='Max length of the encoder input sequence.')
    parser.add_argument(
        '--dropout',
        default=0.1,
        type=float,
        help='')
    parser.add_argument(
        '--use_coordinator',
        action='store_true',
        help=('Whether to use a coordinator to contexualize passages with '
              'other passage vector'))
    parser.add_argument(
        '--coordinator_layers',
        default=1,
        type=int,
        help='Number of hidden layers for the passage coordinator')
    parser.add_argument(
        '--coordinator_heads',
        default=3,
        type=int,
        help='Number of attention heads for the passage coordinator')
    parser.add_argument(
        '--num_token_types',
        default=20,
        type=int,
        help='Number of possiblen token types')
    parser.add_argument(
        '--ignore_token_type',
        default=True,
        type=bool,
        help='Whether to ignore token types or not')
    parser.add_argument(
        '--compute_da_loss',
        action='store_true',
        help='Whether to jointly train dialog act prediction or not')
    parser.add_argument(
        '--decision_function',
        type=int,
        default=1,
        help='Which decision function to use for calculating loss')
    parser.add_argument(
        '--hist_loss_weight',
        type=float,
        default=0.0,
        help='weight of history loss')
    parser.add_argument(
        '--user2agent_loss_weight',
        default=0,
        type=float,
        help=('predict a history agent span based on the previous user '
              'question if > 0'))
    parser.add_argument(
        '--span_marker',
        action='store_true',
        help='mark spans used in history')
    parser.add_argument(
        '--skip_mark_last_user',
        action='store_true',
        help=('skip add mark embeddings of the last user turn to span '
              'embeddings'))
    parser.add_argument(
        '--marker_after_steps',
        default=0,
        type=int,
        help='not using marker in the begining of the training process')
    parser.add_argument(
        '--use_z_attn',
        default=True,
        type=bool,
        help='')


def add_training_params(parser: argparse.ArgumentParser):
    """Common parameters for training."""
    parser.add_argument(
        '--origin_data_dir',
        # default=None,
        default='dataset/Chunyu',
        type=str,
        help='File pattern for the train set.')
    parser.add_argument(
        '--data_dir',
        # default=None,
        default='dataset/Chunyu/processed',
        type=str,
        help='File pattern for the train set.')
    parser.add_argument(
        '--train_file',
        # default=None,
        default='train.json',
        type=str,
        help='File pattern for the train set.')
    parser.add_argument(
        '--dev_file',
        default='dev.json',
        type=str,
        help='File pattern for the dev set.')
    parser.add_argument(
        '--knowledge_file',
        default=None,
        type=str,
        help='File pattern for the dev set.')
    # parser.add_argument(
    #     '--dev_file',
    #     default='dataset/Chunyu/processed/test.json',
    #     type=str,
    #     help='File pattern for the dev set.')
    # parser.add_argument(
    #     '--dev_file',
    #     default='dataset/Chunyu/processed_with_spoken/test.json',
    #     type=str,
    #     help='File pattern for the dev set.')
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='Amount of questions per batch.')
    parser.add_argument(
        '--dev_batch_size',
        type=int,
        default=512,
        help='amount of questions per batch for dev set validation.')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='random seed for initialization and dataset shuffling.')
    parser.add_argument(
        '--adam_eps',
        default=1e-8,
        type=float,
        help='Epsilon for Adam optimizer.')
    parser.add_argument(
        '--adam_betas',
        default='(0.9, 0.999)',
        type=str,
        help='Betas for Adam optimizer.')
    parser.add_argument(
        '--max_grad_norm',
        default=1.0,
        type=float,
        help='Max gradient norm.')
    parser.add_argument(
        '--log_batch_step',
        default=1000,
        type=int,
        help='Number of steps to log during training.')
    parser.add_argument(
        '--train_rolling_loss_step',
        default=1000,
        type=int,
        help='Number of steps of interval to save training loss.')
    parser.add_argument(
        '--weight_decay',
        default=0.01,
        type=float,
        help='Weight decay for optimizer.')
    parser.add_argument(
        '--learning_rate',
        default=2e-5,
        type=float,
        help='Learning rate.')
    parser.add_argument(
        '--warmup_steps',
        default=1000,
        type=int,
        help='Linear warmup over warmup_steps.')
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of update steps to accumulate before updating parameters.')
    parser.add_argument(
        '--num_train_epochs',
        default=100,
        type=float,
        help='Total number of training epochs to perform.')
    parser.add_argument(
        '--auto_resume',
        action='store_true',
        help='Auto resume from latest checkpoint')
    parser.add_argument(
        '--save_checkpoint_every_minutes',
        type=int,
        default=15,
        help='Save a checkpoint every x minutes')
    parser.add_argument(
        '--eval_step',
        default=1000,
        type=int,
        help='Batch steps to run validation and save checkpoint.')
    parser.add_argument(
        '--eval_top_docs',
        # nargs='+',
        type=int,
        default=50,
        help=('Top retrival passages thresholds to analyze prediction results '
              'for'))
    parser.add_argument(
        '--checkpoint_filename_prefix',
        type=str,
        default='dialki',
        help='Checkpoint filename prefix.')

    # parser.add_argument(
    #     '--output_dir',
    #     type=str,
    #     default='dataset/Chunyu/exp_t5_small_chinese_stage1',
    #     help='Output directory for checkpoints.')
    # parser.add_argument(
    #     '--log_dir',
    #     type=str,
    #     default='dataset/Chunyu/exp_t5_small_chinese_stage1/log',
    #     help='Output directory for checkpoints.')
    parser.add_argument(
        '--stage1_index_file',
        type=str,
        default=None,
        help='Output directory for checkpoints.') 

    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset/Chunyu/exp_t5_small_chinese_stage_all',
        help='Output directory for checkpoints.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='dataset/Chunyu/exp_t5_small_chinese_stage_all/log',
        help='Output directory for checkpoints.')

    # parser.add_argument(
    #     '--output_dir',
    #     type=str,
    #     default='dataset/Chunyu/exp_t5_base_chinese',
    #     help='Output directory for checkpoints.')
    # parser.add_argument(
    #     '--log_dir',
    #     type=str,
    #     default='dataset/Chunyu/exp_t5_base_chinese/log',
    #     help='Output directory for checkpoints.')

    parser.add_argument(
        '--model_recover_dir',
        type=str,
        default='dataset/Chunyu/exp_t5_small_chinese_stage_all/model.{}.bin',
        help='Output directory for checkpoints.')
    parser.add_argument(
        '--model_recover_path',
        type=str,
        default=None,
        # default='dataset/Chunyu/exp_t5_small_chinese_stage_all/model.35.bin',
        help='Output directory for checkpoints.')
    parser.add_argument(
        '--inference_only',
        action='store_true',
        default=False,
        help='Inference only.')
    parser.add_argument(
        '--prediction_results_file',
        default='dataset/Chunyu/exp/dev_infer_predictions.json',
        type=str,
        help='Path to a file to write prediction results to')

    parser.add_argument(
        "--add_category",
        action='store_true',
        default=False,
        help="directory path of the output data",
    )
    parser.add_argument(
        "--add_spoken",
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
    parser.add_argument(
        "--rectify",
        action='store_true',
        default=False,
        help="directory path of the output data",
    )
    parser.add_argument(
        "--add_prior_knowledge",
        action='store_true',
        default=False,
        help="directory path of the output data",
    )
    parser.add_argument(
        "--t5_style",
        action='store_true',
        default=False,
        help="directory path of the output data",
    )
    parser.add_argument(
        '--low_resource_flag',
        type=str,
        default=None,
        # default='dataset/Chunyu/exp_t5_small_chinese_stage_all/model.35.bin',
        help='Output directory for checkpoints.')

def add_cuda_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='The parameter for distributed training.')
    parser.add_argument(
        '--fp16',
        default=True,
        type=bool,
        help='Whether to use 16-bit float precision instead of 32-bit.')
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O2',
        help=('For fp16: Apex AMP optimization level selected.'
              'See details at https://nvidia.github.io/apex/amp.html.'))


def get_encoder_checkpoint_params_names():
    return [
        'do_lower_case',
        'pretrained_model_cfg',
        'projection_dim',
        'max_seq_len',
    ]


def get_encoder_params_state(args):
    """
    Selects the param values to be saved in a checkpoint, so that a trained
    model faile can be used for downstream tasks without the need to specify
    these parameter again.

    Return: Dict of params to memorize in a checkpoint.
    """
    params_to_save = get_encoder_checkpoint_params_names()

    r = {}
    for param in params_to_save:
        r[param] = getattr(args, param)
    return r


def set_encoder_params_from_state(state, args):
    if not state:
        return
    params_to_save = get_encoder_checkpoint_params_names()

    override_params = [
        (param, state[param])
        for param in params_to_save
        if param in state and state[param]
    ]
    for param, value in override_params:
        if param == "pretrained_model_cfg":
            continue
        if hasattr(args, param):
            if dist_utils.is_local_master():
                logger.warning(
                    f'Overriding args parameter value from checkpoint state. '
                    f'{param = }, {value = }')
        setattr(args, param, value)
    return args


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_args_gpu(args):
    """
    Setup arguments CUDA, GPU & distributed training.
    """

    world_size = os.environ.get('WORLD_SIZE')
    world_size = int(world_size) if world_size else 1
    args.distributed_world_size = world_size
    local_rank = args.local_rank
  
    if local_rank == -1:
        # Single-node multi-gpu (or cpu) mode.
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        device = torch.device(device)
        n_gpu = args.n_gpu = torch.cuda.device_count()
    else: 
        # Distributed mode.
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        # set up the master's ip address so this child process can coordinate
        torch.distributed.init_process_group(
            backend='nccl',
            rank=args.local_rank,
            world_size=world_size)
        n_gpu = args.n_gpu = 1
    args.device = device

    if dist_utils.is_local_master():
        logger.info(
            f'Initialized host {socket.gethostname()}'
            f'{local_rank = } {device = } {n_gpu = } {world_size = }'
            f'16-bits training: {args.fp16}')
