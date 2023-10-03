import numpy as np
from functools import partial
import json

def _evaluate_count_empty(pred_labels, gold_labels):
    if len(pred_labels) == 0:
        pred_labels.add('empty')
    if len(gold_labels) == 0:
        gold_labels.add('empty')
    tp = len(pred_labels & gold_labels)
    r = tp / len(gold_labels)
    p = tp / len(pred_labels)
    try:
        f1 = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1 = 0
    return p, r, f1

def _evaluate_notcount_empty(pred_labels, gold_labels):
    tp = len(pred_labels & gold_labels)
    try:
        r = tp / len(gold_labels)
        p = tp / len(pred_labels)
        f1 = 2 * p * r / (p + r)
    except ZeroDivisionError:
        p = 0
        r = 0
        f1 = 0
    return p, r, f1

def _evaluate(pred_labels, gold_labels, count_empty):
    if count_empty:
        return _evaluate_count_empty(pred_labels, gold_labels)
    else:
        return _evaluate_notcount_empty(pred_labels, gold_labels)
    

def _construct_prefixs(labels):
    prefixs = dict()
    for label in labels:
        prefix, status = label.split('-')
        status = status.split(':')[-1]
        try:
            prefixs[prefix].add(status)
        except KeyError:
            prefixs[prefix] = {status}
    return prefixs

def _merge(previous_statuses_w, current_statuses_w):
    if '阳性' in previous_statuses_w and '阴性' in current_statuses_w:
        previous_statuses_w.remove('阳性')
    if '阴性' in previous_statuses_w and '阳性' in current_statuses_w:
        previous_statuses_w.remove('阴性')
    if '医生阳性' in previous_statuses_w and '医生阴性' in current_statuses_w:
        previous_statuses_w.remove('医生阳性')
    if '医生阴性' in previous_statuses_w and '医生阳性' in current_statuses_w:
        previous_statuses_w.remove('医生阴性')
    if '未知' in previous_statuses_w and len(current_statuses_w) > 0:
        previous_statuses_w.remove('未知')
    if len(previous_statuses_w) > 0 and '未知' in current_statuses_w:
        current_statuses_w.remove('未知')
    merged_statuses_w = previous_statuses_w | current_statuses_w
    return merged_statuses_w
    
def merge(previous_labels_w, current_labels_w):
    previous_prefixs = _construct_prefixs(previous_labels_w)
    current_prefixs = _construct_prefixs(current_labels_w)
    for key in current_prefixs.keys():
        if key not in previous_prefixs:
            previous_prefixs[key] = current_prefixs[key]
        else:
            previous_prefixs[key] = _merge(previous_prefixs[key], current_prefixs[key])
    merged_labels_w = set()
    for key in previous_prefixs.keys():
        for status in previous_prefixs[key]:
            merged_labels_w.add('{}-{}'.format(key, status))
    return merged_labels_w

def _window_eval(window_pred_labels_w, window_gold_labels_w, count_empty, func):
    ps = []
    rs = []
    f1s = []
    for pred_label_w, gold_label_w in zip(window_pred_labels_w, window_gold_labels_w):
        pred_label_w = set(map(func, pred_label_w))
        gold_label_w = set(map(func, gold_label_w))
        p, r, f1 = _evaluate(pred_label_w, gold_label_w, count_empty)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    p = sum(ps) / len(ps)
    r = sum(rs) / len(rs)
    f1 = sum(f1s) / len(f1s)
    infos = {
        'p': p,
        'r': r,
        'f1': f1
    }
    return infos


def _dialog_eval(window_pred_labels_w, window_gold_labels_w, dialogs, count_empty, func):
    i = 0
    ps = []
    rs = []
    f1s = []
    for dialog in dialogs:
        dialog_pred_labels_w = set()
        dialog_gold_labels_w = set()
        for window in dialog:
            dialog_pred_labels_w = merge(dialog_pred_labels_w, set(window_pred_labels_w[i]))
            dialog_gold_labels_w = merge(dialog_gold_labels_w, set(window_gold_labels_w[i]))
            i += 1

        dialog_pred_labels_w = set(map(func, dialog_pred_labels_w))
        dialog_gold_labels_w = set(map(func, dialog_gold_labels_w))

        p, r, f1 = _evaluate(dialog_pred_labels_w, dialog_gold_labels_w, count_empty)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    
    p = sum(ps) / len(ps)
    r = sum(rs) / len(rs)
    f1 = sum(f1s) / len(f1s)

    infos = {
        'p': p,
        'r': r,
        'f1': f1
    }
    return infos

_get_category = lambda x: x.split('-')[0].split(':')[0]
_get_item = lambda x: x.split('-')[0]
_get_full = lambda x: x

window_category = partial(_window_eval, func=_get_category)
window_item = partial(_window_eval, func=_get_item)
window_full = partial(_window_eval, func=_get_full)

dialog_category = partial(_dialog_eval, func=_get_category)
dialog_item = partial(_dialog_eval, func=_get_item)
dialog_full = partial(_dialog_eval, func=_get_full)

def label_format_convert(label):
    labels = []
    for key, value in label.items():
        labels.append('{}-状态:{}'.format(key, value))
    return labels

def evaluate_for_file(eval_file='./data/pred_test_after_state.json', count_empty=True):

    with open(eval_file, 'r', encoding='utf8') as f:
        dialogs = json.load(f)
    
    window_pred_labels_w = []
    window_gold_labels_w = []
    for dialog in dialogs:
        for window in dialog:
            window_pred_labels_w.append(window["pred"])
            window_gold_labels_w.append(window["label"])
            # window_pred_labels_w.append(label_format_convert(window['pred']))
            # window_gold_labels_w.append(label_format_convert(window["label"]))

    infos = {
        'window': {},
        'dialog': {}
    }
    infos['window']['category'] = window_category(window_pred_labels_w, window_gold_labels_w, count_empty)
    infos['window']['item'] = window_item(window_pred_labels_w, window_gold_labels_w, count_empty)
    infos['window']['full'] = window_full(window_pred_labels_w, window_gold_labels_w, count_empty)


    infos['dialog']['category'] = dialog_category(window_pred_labels_w, window_gold_labels_w, dialogs, count_empty)
    infos['dialog']['item'] = dialog_item(window_pred_labels_w, window_gold_labels_w, dialogs, count_empty)
    infos['dialog']['full'] = dialog_full(window_pred_labels_w, window_gold_labels_w, dialogs, count_empty)

    # return infos['window']
    return infos['dialog']

def evaluate_for_file_category(eval_file='./data/pred_test_after_state.json', count_empty=True):

    with open(eval_file, 'r', encoding='utf8') as f:
        dialogs = json.load(f)
    
    window_pred_labels_w = []
    window_gold_labels_w = []
    for dialog in dialogs:
        for window in dialog:
            window_pred_labels_w.append(window["pred"])
            window_gold_labels_w.append(window["label"])
            # window_pred_labels_w.append(label_format_convert(window['pred']))
            # window_gold_labels_w.append(label_format_convert(window["label"]))
    categories = ['症状', '检查', '手术', '一般信息']
    window_pred_labels_w_category, window_gold_labels_w_category = {}, {}

    all_infos = {}
    for category in categories:
        window_pred_labels_w_category[category] = window_pred_labels_w + []
        window_gold_labels_w_category[category] = window_gold_labels_w + []
    
        for i in range(len(window_gold_labels_w_category[category])):
            window_gold_labels_w_category[category][i] = [label for label in window_gold_labels_w_category[category][i] if category in label]
            window_pred_labels_w_category[category][i] = [label for label in window_pred_labels_w_category[category][i] if category in label]
    
        infos = {
            'window': {}
        }
        infos['window']['category'] = window_category(window_pred_labels_w_category[category], window_gold_labels_w_category[category], count_empty)
        infos['window']['item'] = window_item(window_pred_labels_w_category[category], window_gold_labels_w_category[category], count_empty)
        infos['window']['full'] = window_full(window_pred_labels_w_category[category], window_gold_labels_w_category[category], count_empty)

        all_infos[category] = infos

    return all_infos


def evaluate_for_file_term_old(eval_file='./data/pred_test_after_state.json', count_empty=True):

    with open(eval_file, 'r', encoding='utf8') as f:
        dialogs = json.load(f)
    
    window_pred_labels_w = []
    window_gold_labels_w = []
    for dialog in dialogs:
        for window in dialog:
            window_pred_labels_w.append(window["pred"])
            window_gold_labels_w.append(window["label"])
            # window_pred_labels_w.append(label_format_convert(window['pred']))
            # window_gold_labels_w.append(label_format_convert(window["label"]))
    window_pred_labels_w_term, window_gold_labels_w_term = [], []
    for i in range(30):
        window_pred_labels_w_term.append([])
        window_gold_labels_w_term.append([])
    for i, label in enumerate(window_gold_labels_w):
        window_gold_labels_w_term[len(label)].append(label)
        window_pred_labels_w_term[len(label)].append(window_pred_labels_w[i])

    all_infos = {}
    for i in range(len(window_gold_labels_w_term)):
        if len(window_gold_labels_w_term[i]) == 0:
            continue
    
        infos = {
            'window': {}
        }
        # infos['window']['category'], ps, rs, f1s = window_category(window_pred_labels_w_term[i], window_gold_labels_w_term[i], count_empty)
        infos['window']['item'] = window_item(window_pred_labels_w_term[i], window_gold_labels_w_term[i], count_empty)
        infos['window']['full'] = window_full(window_pred_labels_w_term[i], window_gold_labels_w_term[i], count_empty)

        all_infos[str(i)] = infos


    return all_infos


def evaluate_for_file_status(eval_file='./data/pred_test_after_state.json', count_empty=True):

    with open(eval_file, 'r', encoding='utf8') as f:
        dialogs = json.load(f)

    with open('dataset/Chunyu/test_status_change.json', 'r', encoding='utf8') as f:
        status_dialogs = json.load(f)
    example_ids = []
    for dialog in status_dialogs:
        for window in dialog:
            example_ids.append([window['dialogue_id'], window['window_id']])
    
    window_pred_labels_w = []
    window_gold_labels_w = []
    for i, dialog in enumerate(dialogs):
        for j, window in enumerate(dialog):
            if [i,j] in example_ids:
                window_pred_labels_w.append(window["pred"])
                window_gold_labels_w.append(window["label"])
            # window_pred_labels_w.append(label_format_convert(window['pred']))
            # window_gold_labels_w.append(label_format_convert(window["label"]))

    infos = {
        'window': {},
        'dialog': {}
    }
    infos['window']['category'] = window_category(window_pred_labels_w, window_gold_labels_w, count_empty)
    infos['window']['item'] = window_item(window_pred_labels_w, window_gold_labels_w, count_empty)
    infos['window']['full'] = window_full(window_pred_labels_w, window_gold_labels_w, count_empty)

    return infos['window']



def evaluate_for_file_term(eval_file='./data/pred_test_after_state.json', count_empty=True):

    with open(eval_file, 'r', encoding='utf8') as f:
        dialogs = json.load(f)
    
    window_pred_labels_w = []
    window_gold_labels_w = []
    for dialog in dialogs:
        for window in dialog:
            window_pred_labels_w.append(window["pred"])
            window_gold_labels_w.append(window["label"])
            # window_pred_labels_w.append(label_format_convert(window['pred']))
            # window_gold_labels_w.append(label_format_convert(window["label"]))
    window_pred_labels_w_term, window_gold_labels_w_term = [], []
    for i in range(4):
        window_pred_labels_w_term.append([])
        window_gold_labels_w_term.append([])
    for i, label in enumerate(window_gold_labels_w):
        term_num = len(label)
        if term_num == 0:
            index = 0
        elif term_num <= 4:
            index = 1
        elif term_num <= 8:
            index = 2
        else:
            index = 3
        window_gold_labels_w_term[index].append(label)
        window_pred_labels_w_term[index].append(window_pred_labels_w[i])

    all_infos = {}
    for i in range(len(window_gold_labels_w_term)):
        if len(window_gold_labels_w_term[i]) == 0:
            continue
    
        infos = {
            'window': {}
        }
        # infos['window']['category'], ps, rs, f1s = window_category(window_pred_labels_w_term[i], window_gold_labels_w_term[i], count_empty)
        infos['window']['item'] = window_item(window_pred_labels_w_term[i], window_gold_labels_w_term[i], count_empty)
        infos['window']['full'] = window_full(window_pred_labels_w_term[i], window_gold_labels_w_term[i], count_empty)

        all_infos[str(i)] = infos


    return all_infos

def main_old():

    eval_files = ['data/pred_test_MIE_multi.json', 'data/pred_test_MIE_multi_after_state.json', 
                'data/pred_test_after_state.json', 'data/gold_test_after_state.json']
    

    with open('data/test.json', 'r', encoding='utf8') as f:
        all_dialogs = json.load(f)

    ps_MIE, rs_MIE, f1s_MIE, dialogs_MIE = evaluate_for_file(eval_file='data/pred_test_MIE_multi.json')
    ps_MIE_as_input, rs_MIE_as_input, f1s_MIE_as_input, dialogs_MIE_as_input = evaluate_for_file(eval_file='data/pred_test_MIE_multi_after_state.json')
    ps_ours, rs_ours, f1s_ours, dialogs_ours = evaluate_for_file(eval_file='data/pred_test_after_state.json')
    ps_gold_as_input, rs_gold_as_input, f1s_gold_as_input, dialogs_gold_as_input = evaluate_for_file(eval_file='data/gold_test_after_state.json')

    index = 0
    for i, dialog in enumerate(all_dialogs):
        for j, window in enumerate(dialog):
            window['MIE'] = dialogs_MIE[i][j]['pred_entity']
            window['MIE_as_input'] = dialogs_MIE_as_input[i][j]['pred_entity']
            window['ours'] = dialogs_ours[i][j]['pred_entity']
            window['gold_as_input'] = dialogs_gold_as_input[i][j]['pred_entity']

            window['description'] = ''
            if f1s_MIE_as_input[index] > f1s_MIE[index]:
                window['description'] += '我们状态模块更好 '
            
            if f1s_ours[index] > f1s_MIE_as_input[index]:
                window['description'] += '我们实体模块更好 '

            if f1s_gold_as_input[index] > f1s_ours[index]:
                window['description'] += '实体抽取错误 '

            if f1s_gold_as_input[index] < 1:
                window['description'] += '状态抽取错误 '

            index += 1
    
    with open('log/output_results/all_pred.json', 'w', encoding='utf-8') as f:
        json.dump(all_dialogs, f, indent=2, ensure_ascii=False)


def main():
    # Chunyu
    # eval_files = ['dataset/Chunyu/exp_t5_base_chinese_v7_add_category_add_state/pred_stage_all_test_97_two_stages.json']
    # eval_files = ['dataset/Chunyu/exp_t5_base_chinese_one_stage_v2/pred_stage_all_test_218_one_stage.json']
    # eval_files = ['dataset/Chunyu/exp_t5_base_chinese_v7_add_category/pred_stage_all_test_97_two_stages.json']
    # eval_files = ['dataset/Chunyu/exp_t5_base_chinese_v5_add_state/pred_stage_all_test_92_two_stages.json']
    # eval_files = ['dataset/Chunyu/exp_t5_base_chinese_v5/pred_stage_all_test_71_two_stages.json']

    # eval_files = ['dataset/Chunyu/exp_t5_small_chinese_one_stage_v2/pred_stage_all_test_226_one_stage.json']
    # eval_files = ['dataset/Chunyu/exp_t5_small_chinese_v7_add_category_add_state/pred_stage_all_test_96_two_stages.json']

    # CMDD
    # eval_files = ['dataset/CMDD/exp_t5_small_chinese_one_stage_v2/pred_stage_all_test_71_one_stage.json']
    # eval_files = ['dataset/CMDD/exp_t5_base_chinese_one_stage_v2/pred_stage_all_test_185_one_stage.json']
    # eval_files = ['dataset/CMDD/exp_t5_base_chinese_v5/pred_stage_all_test_27_two_stages.json']
    # eval_files = ['dataset/CMDD/exp_t5_base_chinese_v7_add_state/pred_stage_all_test_27_two_stages.json']
    eval_files = ['dataset/CMDD/exp_t5_small_chinese_v7_add_category_add_state/pred_stage_all_test_47_two_stages.json']
    
    infos = evaluate_for_file(eval_file=eval_files[0], count_empty=True)

    # all_infos = evaluate_for_file_category(eval_file=eval_files[0], count_empty=False)
    # infos = evaluate_for_file(eval_file=eval_files[0], count_empty=False)
    # infos = evaluate_for_file_term(eval_file=eval_files[0], count_empty=True)
    # infos = evaluate_for_file_status(eval_file=eval_files[0], count_empty=True)

    print('a')


if __name__ == "__main__":
    """
    将所有模型预测的结果合并在一起对比，进行case study
    """
    main()