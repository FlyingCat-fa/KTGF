import collections

import torch
import numpy as np
import torch.nn.utils.rnn as rnn_utils



def _pad(target, fill_value, pad_len, dim=0):
    if pad_len == 0:
        return target
    size = list(target.size())
    size[dim] = pad_len
    pad = torch.full(size, fill_value)
    return torch.cat([target, pad], dim=dim)


def collate_fn(batch):

    input_ids, attention_masks, labels, dial_id,  window_id, term_id = list(zip(*batch))
        
    input_ids = [torch.tensor(instance) for i, instance in enumerate(input_ids)]
    attention_masks = [torch.tensor(instance) for i, instance in enumerate(attention_masks)]
    labels = [torch.tensor(instance) for i, instance in enumerate(labels)]

    dial_id = torch.tensor(dial_id)
    window_id = torch.tensor(window_id)
    term_id = torch.tensor(term_id)

    input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0) # pad_id
    input_masks = rnn_utils.pad_sequence(attention_masks, batch_first=True, padding_value=0) # pad_id
    labels = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=-100)

    return input_ids, input_masks, labels, dial_id,  window_id, term_id