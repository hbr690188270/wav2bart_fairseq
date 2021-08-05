import torch
import numpy as np
from fairseq.tasks.asr_finetuning_v3 import ASRFinetunig_v3, ASRFinetuningConfig
from fairseq.models.wav2bart.wav2bart import Wav2Bart, Wav2BartConfig
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar
import pickle

# model_config = Wav2BartConfig
# task_config = ASRFinetuningConfig

# model_config.w2v_path = '/data/bairu/model_cache/wav2vec_model/wav2vec_small.pt'
# model_config.bart_path = '/data/bairu/model_cache/bart_model/bart.base/'
# task_config.bart_path = '/data/bairu/model_cache/bart_model/bart.base/'
# checkpoint_dir = '/data/bairu/repos/wav2bart_fairseq/outputs/2021-07-15/05-09-30/checkpoints'
# restore_file = 'checkpoint_best.pt'
use_fp16 = False
use_cuda = True


# task = ASRFinetunig_v3.setup_task(task_config)
# model = task.build_model(model_config)

# model_path = '/data/bairu/repos/wav2bart_fairseq/outputs/2021-07-15/05-09-30/checkpoints/checkpoint_best.pt'
# model_path = '/data/bairu/repos/wav2bart_fairseq/outputs/2021-07-29/12-11-17/checkpoints/checkpoint_last.pt'

# bart v1
# model_path = '/data/bairu/repos/wav2bart_fairseq/outputs/2021-07-15/05-09-30/checkpoints/checkpoint_best.pt'

# bart v2
# model_path = '/data/bairu/repos/wav2bart_fairseq/examples/wav2bart/config/outputs/2021-07-31/15-48-35/checkpoints/checkpoint_best.pt'

# bart rand all
# model_path = '/data/bairu/repos/wav2bart_fairseq/outputs/2021-08-03/16-35-05/checkpoints/checkpoint_best.pt'

## gpt2 train 10000
model_path = '/data/bairu/repos/wav2bart_fairseq/outputs/2021-08-01/16-58-36/checkpoints/checkpoint_last.pt'

## gpt2 train1
# model_path = '/data/bairu/repos/wav2bart_fairseq/outputs/2021-08-01/10-35-25/checkpoints/checkpoint_last.pt'

## gpt2 train100
# model_path = '/data/bairu/repos/wav2bart_fairseq/outputs/2021-08-04/14-40-19/checkpoints/checkpoint_last.pt'

## gpt2 random train100
# model_path = '/data/bairu/repos/wav2bart_fairseq/outputs/2021-08-04/15-34-23/checkpoints/checkpoint_last.pt'



models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
)
model = models[0]

for model in models:
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()


task.cfg.eval_wer = False
saved_cfg.criterion._name = "cross_entropy_with_acc"
DATASET = 'train_10000'


criterion = task.build_criterion(saved_cfg.criterion)
criterion.eval()



task.load_dataset(DATASET, combine=False, epoch=1, task_cfg=saved_cfg.task)
dataset = task.dataset(DATASET)

# Initialize data iterator
itr = task.get_batch_iterator(
    dataset=dataset,
    max_tokens=saved_cfg.dataset.max_tokens,
    max_sentences=saved_cfg.dataset.batch_size,
    max_positions=utils.resolve_max_positions(
        task.max_positions(),
        *[m.max_positions() for m in models],
    ),
    ignore_invalid_inputs=saved_cfg.dataset.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=saved_cfg.dataset.required_batch_size_multiple,
    seed=saved_cfg.common.seed,
    num_shards=1,
    shard_id=0,
    num_workers=saved_cfg.dataset.num_workers,
    data_buffer_size=saved_cfg.dataset.data_buffer_size,
).next_epoch_itr(shuffle=True)

progress = progress_bar.progress_bar(
    itr,
    log_format=saved_cfg.common.log_format,
    log_interval=saved_cfg.common.log_interval,
    prefix=f"valid on dev subset",
    default_log_format=("tqdm" if not saved_cfg.common.no_progress_bar else "simple"),
)


def debug_attention(task, model, progress,):
    decode_res = []
    for i, sample in enumerate(progress):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        print(sample.keys())
        print(sample['net_input']['source'].size())
        # print(sample['target'])
        # print([len(x) for x in sample['target']])
        net_output = model(**sample['net_input'])
        attn = net_output[1]['attn'][0]
        print(attn.size())

        for idx in range(len(sample['target'])):
            data_dict = {}
            curr_target = utils.strip_pad(sample["target"][idx], task.target_dictionary.pad())
            curr_length = len(curr_target)
            curr_tokens = task.bart.decode(curr_target)
            print(curr_target)
            print(curr_length)
            print(curr_tokens)
            curr_att = attn[idx]
            print(curr_att.size())
            data_dict['target_tensor'] = curr_target.int().cpu().numpy()
            data_dict['target_length'] = curr_length
            data_dict['target_tokens'] = curr_tokens
            data_dict['attention'] = curr_att.float().detach().cpu().numpy()

            decode_res.append(data_dict)
        break

    with open("decode_res.pkl",'wb') as f:
        pickle.dump(decode_res, f)


def check_tgt_dict(task, model):
    tgt_dict = task.target_dictionary
    # print(tgt_dict.indices)
    print(len(tgt_dict.indices))
    print(model.decoder.decoder.embed_tokens.weight.size())
    bpe = task.bart.bpe
    print(bpe.bpe.encoder)
    print(len(bpe.bpe.encoder))
    # print(bpe.bpe.bype_encoder)


def validate(task, model, progress):
    log_outputs = []
    total_len = len(progress)
    for i, sample in enumerate(progress):
        # if i > 200:
            # break
        print(i,"/", total_len)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
        progress.log(log_output, step=i)
        log_outputs.append(log_output)

    with metrics.aggregate() as agg:
        task.reduce_metrics(log_outputs, criterion)
        log_output = agg.get_smoothed_values()

    print(log_output)
    progress.print(log_output, tag='dev', step=i)

def test_generate(task, model, progress):
    log_outputs = []
    total_len = len(progress)
    for i, sample in enumerate(progress):
        if i > 10:
            break
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
        progress.log(log_output, step=i)
        log_outputs.append(log_output)

    with metrics.aggregate() as agg:
        task.reduce_metrics(log_outputs, criterion)
        log_output = agg.get_smoothed_values()

    print(log_output)
    progress.print(log_output, tag='dev', step=i)


if __name__ == '__main__':
    # check_tgt_dict(task, model)
    # validate(task, model, progress)
    test_generate(task, model, progress)

