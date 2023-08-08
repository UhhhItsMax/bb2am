# BlenderBot
## Command
1. **BB3**:
    - Interactive
```bash
parlai safe_interactive --model-file zoo:bb3/bb3_3B/model --init-opt gen/r2c2_bb3 --search_server 0.0.0.0:8080 --loglevel debug --verbose  
```
2. **BB2**:
    - Interactive
```bash
python -m parlai interactive --model-file zoo:blenderbot2/blenderbot2_400M/model --search_server 0.0.0.0:8080 --loglevel debug
```
3. **SearchServer**:
```bash
python search_server.py serve --host 0.0.0.0:8080
```
## Environment
parlai1

## Error Collection
### When using display_data command: `parlai display_data --task hotpotqa`
- Error message: pkg_resources.DistributionNotFound: The 'traitlets>=5' distribution was not found and is required by ipython
- Solution: reinstall ipython `pip install ipython`

### When training BB2 with hotpotqa got error: `KeyError: 'accuracy'`
- Solution: add `--vmm/--validation-metric min`, default is `accuracy`. With `min` is `'loss', 'ppl', 'mean_rank'`
- resource
    * https://github.com/facebookresearch/ParlAI/issues/1927
### AssertionError: Checksum for ed_persona_topicifier__train__experiencer_only.json 
from 
http://parl.ai/downloads/blended_skill_talk/ed_persona_topicifier__train__experiencer_only.json
does not match the expected checksum:

## Experiment
### Metrics explaniation
#### [Agent-specific metrics](https://parl.ai/docs/tutorial_metrics.html#agent-specific-metrics)
- `ppl` and `token_acc` : the perplexity and per-token accuracy. these are generative performance metrics.
- `tpb`, `ctpb`, `ltpb`: stand for tokens per batch, context-tokens per batch, and label-tokens per batch. These are useful for measuring how dense the batches are, and are helpful when experimenting with dynamic batching. tpb is always the sum of ctpb and lptb.
- `tps`, `ctps`, `ltps`: are similar, but stand for “tokens per second”. They measure how fast we are training. Similarly, exps measures examples per second.
- `gpu_mem`: measures roughly how much GPU memory your model is using, but it is only approximate. This is useful for determining if you can possibly increase the model size or the batch size.
- `loss`: the loss metric

### Training hotpotqa following this issue: https://github.com/facebookresearch/ParlAI/issues/4347
```bash
parlai train_model -dp data \
--model projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent \
--task hotpotqa --num_epochs 1 \
--memory-decoder-model-file "" --memory-key personas \
--search-query-generator-model-file zoo:blenderbot2/query_generator/model --search-query-generator-beam-min-length 2 \
--save-every-n-secs 100 --validation_every_n_secs 100 --log_every_n_secs 60 \
--init-model zoo:blenderbot2/blenderbot2_400M/model --dict-file zoo:blenderbot2/blenderbot2_400M/model.dict \
--datatype train:stream  \
--embeddings-scale True --variant prelayernorm --split-lines True --learn-positional-embeddings True \
--n-layers 12 --embedding-size 1024 --ffn-size 4096 --n-heads 16 --n-decoder-layers 12 \
--dict-tokenizer gpt2 --generation-model bart \
--query-model bert_from_parlai_rag \
--rag-model-type token --rag-retriever-type search_engine --search_server None \
--dpr-model-file zoo:hallucination/bart_rag_token/model \
--gold-document-titles-key select-docs-titles --insert-gold-docs True \
--beam-min-length 20 --beam-context-block-ngram 3 --beam-block-ngram 3 --beam-block-full-context False --beam-size 10 \
--inference beam --optimizer mem_eff_adam --learningrate 1e-05 --lr-scheduler-patience 1 --model-parallel True \
--knowledge-access-method memory_only \
--batchsize 1 \
--truncate 512 --text-truncate 512 --label-truncate 128 \
--dropout 0.0 --attention-dropout 0.0 \
--min-doc-token-length 64 --max-doc-token-length 256 \
--fp16 True --fp16-impl mem_efficient --force-fp16-tokens True \
--tensorboard-log true --model-file /content/ParlAI/tmp/model \
--eval-batchsize 2 --skip-generation true\
--vmt ppl -vmm min
```
With this command, it is training with following output:
```bash
12:17:09 | training...
12:18:15 | time:65s total_exs:7 total_steps:7 epochs:0.00 time_left:834949s
    clen  clip  ctpb  ctps  ctrunc  ctrunclen  exps  exs  gnorm  llen  loss    lr  ltpb  ltps  ltrunc  ltrunclen   ppl  token_acc  token_em  total_train_updates   tpb   tps   ups
    1331     1   512 55.48       1      818.7 .1084    7  33.74 3.286 2.465 1e-05 3.286 .3560       0          0 11.76      .5652     .1429                    7 515.3 55.83 .1084

12:19:22 | time:132s total_exs:14 total_steps:14 epochs:0.00 time_left:854819s
    clen  clip  ctpb  ctps  ctrunc  ctrunclen  exps  exs  gnorm  llen  loss    lr  ltpb  ltps  ltrunc  ltrunclen   ppl  token_acc  token_em  total_train_updates   tpb   tps   ups
    1440     1   512 53.99       1      928.4 .1055    7  39.18 4.571 3.288 1e-05 4.571 .4821       0          0 26.79      .4375         0                   14 516.6 54.48 .1055

12:20:28 | time:199s total_exs:21 total_steps:21 epochs:0.00 time_left:857920s
    clen  clip  ctpb  ctps  ctrunc  ctrunclen  exps  exs  gnorm  llen  loss    lr  ltpb  ltps  ltrunc  ltrunclen   ppl  token_acc  token_em  total_train_updates   tpb   tps   ups
    1324     1   512 53.59       1      811.7 .1047    7  22.26 5.429 1.243 1e-05 5.429 .5682       0          0 3.466      .6842     .1429                   21 517.4 54.16 .1047

12:21:29 | time:260s total_exs:27 total_steps:27 epochs:0.00 time_left:870269s
    clen  clip  ctpb  ctps  ctrunc  ctrunclen  exps  exs  gnorm  llen  loss    lr  ltpb  ltps  ltrunc  ltrunclen   ppl  token_acc  token_em  total_train_updates   tpb   tps   ups
    1502     1   512 50.69       1      989.8 .0990    6  22.12 4.333 1.882 1e-05 4.333 .4290       0          0 6.568      .6923     .1667                   27 516.3 51.11 .0990

12:22:39 | time:329s total_exs:34 total_steps:34 epochs:0.00 time_left:875759s
    clen  clip  ctpb  ctps  ctrunc  ctrunclen  exps  exs  gnorm  llen  loss    lr  ltpb  ltps  ltrunc  ltrunclen   ppl  token_acc  token_em  total_train_updates  tpb   tps   ups
    1369     1   512 51.62       1      857.4 .1008    7  27.46     3 1.174 1e-05     3 .3024       0          0 3.235      .7143     .1429                   34  515 51.92 .1008
```

### Some command options
#### For log
Tensorboard Arguments:  
  --tensorboard-log, --tblog TENSORBOARD_LOG  
        Tensorboard logging of metrics (default: False)  
  --tensorboard-logdir, --tblogdir TENSORBOARD_LOGDIR  
        Tensorboard logging directory, defaults to model_file.tensorboard (default: None)

## Useful command
### For datasets statics
```bash
parlai data_stats --task hotpotqa
# --- hotpotqa ---
# loaded 90447 episodes with a total of 90447 examples
#            avg_utterance_length    tokens  unique_tokens  unique_utterances  utterances
#    both                     604 109264534         645571             143683      180894
#    input                   1206 109036813         645569              90447       90447
#    labels                 2.518    227721          38142              53236       90447

# --- msc ---
# loaded 35880 episodes with a total of 236987 examples
#                      avg_utterance_length    exs   tokens  unique_tokens  unique_utterances  utterances
#    both                             37.57        17805097          43746             265868      473974
#    input                            55.53        13160412          42999             234030      236987
#    labels                            19.6         4644685          39707             229906      236987
#    msc:Session1Self                       131438
#    msc_dialogue_2                          46420
#    msc_dialogue_3                          47259
#    msc_dialogue_4                          11870

# --- blended_skill_talk ---
# loaded 4819 episodes with a total of 27018 examples
#            avg_utterance_length  tokens  unique_tokens  unique_utterances  utterances
#    both                   21.22 1146458          25341              53289       54036
#    input                     26  702433          19882              26904       27018
#    labels                 16.43  444025          16590              26475       27018

# --- coqa ---
# loaded 7199 episodes with a total of 108647 examples
#            avg_utterance_length  tokens  unique_tokens  unique_utterances  utterances
#    both                   16.15 3508678          74411             153822      217294
#    input                  29.27 3180619          72865              88211      108647
#    labels                 3.019  328059          36514              65621      108647
```
## Datasets
| Dataset            | Describe                                                                                 | Examples | Used |
| ------------------ | ---------------------------------------------------------------------------------------- | -------- | ---- |
| Hotpotqa           | multi-hop question answering                                                             | 90447    | yes  |
| Msc                | A multi-session human-human chit-chat dataset                                            | 236987   | no   |
| blended_skill_talk | A dataset of 7k conversations explicitly designed to exhibit multiple conversation modes | 27018    | yes  |
| coqa               | CoQA is a large-scale dataset for building Conversational Question Answering systems     | 108647   | yes  |

# Experiments log
## 03.08.2023
### Try with better GPU to see the validation speed
- Using V100 for validation hotpotsq: without generation 670s, with generation over 20min still no results
- Using A100 didn't improve the speed, maybe because the batch size is still 1
### New training command
```bash
parlai train_model -dp data \
--model projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent \
--task hotpotqa --num_epochs 1 \
--memory-decoder-model-file "" --knowledge-access-method memory_only --memory-key full_text \ 
# change memory-key to full_text
--search-query-generator-model-file zoo:blenderbot2/query_generator/model --search-query-generator-beam-min-length 2 \
--save-every-n-secs 100 --validation_every_n_secs 100 --log_every_n_secs 60 \
--init-model zoo:blenderbot2/blenderbot2_400M/model --dict-file zoo:blenderbot2/blenderbot2_400M/model.dict \
--datatype train:stream  \
--embeddings-scale True --variant prelayernorm --split-lines True --learn-positional-embeddings True \
--n-layers 12 --embedding-size 1024 --ffn-size 4096 --n-heads 16 --n-decoder-layers 12 \
--dict-tokenizer gpt2 --generation-model bart \
--query-model bert_from_parlai_rag \
--rag-model-type token --rag-retriever-type search_engine --search_server None \
--dpr-model-file zoo:hallucination/bart_rag_token/model \
--gold-document-titles-key select-docs-titles --insert-gold-docs True \
--beam-min-length 20 --beam-context-block-ngram 3 --beam-block-ngram 3 --beam-block-full-context False --beam-size 10 \
--inference beam --optimizer mem_eff_adam --learningrate 1e-05 --lr-scheduler-patience 1 --model-parallel True \
--batchsize 1 \
--truncate 512 --text-truncate 512 --label-truncate 128 \
--dropout 0.0 --attention-dropout 0.0 \
--min-doc-token-length 64 --max-doc-token-length 256 \
--fp16 True --fp16-impl mem_efficient --force-fp16-tokens True \
--tensorboard-log true --model-file /content/gdrive/MyDrive/BB2AM/model \
# save model to GDrive
--eval-batchsize 2 --skip-generation true\
--vmt ppl -vmm min
```
current result
```bash
best ppl 11.8, loss 2.468, tocken_acc 0.5648
```

## 04.08.2023
### Next step
- [x] Try with partly trained model 
- [x] Try with other datasets
- [x] Try multi task training, see this [issue](https://github.com/facebookresearch/ParlAI/issues/4391)
```
--multitask-weights 4,3,3 \
--task wizard_of_internet,msc,custom_dataset \
```
- [x] Adjust the validation to after how many steps and same as save 
  
### training with msc, wio, hotpotqa, coqa
Validation takes 2151.11s, estimated training time is 56h


### training with blender_talk_skill, hotpotqa, coqa
Validation takes 1169.71s, estimated training time is 100h ( it is not correct because it's using the model which was tried out with msc, wio ect.)
After clean start, estimated traing time is 16h

### New training command
```bash
parlai train_model -dp data \
--model projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent \
--task blended_skill_talk,hotpotqa,coqa \
--num_epochs 1 \
--memory-decoder-model-file "" --knowledge-access-method memory_only --memory-key full_text \
--search-query-generator-model-file zoo:blenderbot2/query_generator/model --search-query-generator-beam-min-length 2 \
--save-after-valid True --vstep 1000 --lstep 100 --tstep 10000 -vp 5 \ 
# modify validation, logstep, total test steps and inpatient times
--init-model zoo:blenderbot2/blenderbot2_400M/model --dict-file zoo:blenderbot2/blenderbot2_400M/model.dict \
--datatype train:stream  \
--embeddings-scale True --variant prelayernorm --split-lines True --learn-positional-embeddings True \
--n-layers 12 --embedding-size 1024 --ffn-size 4096 --n-heads 16 --n-decoder-layers 12 \
--dict-tokenizer gpt2 --generation-model bart \
--query-model bert_from_parlai_rag \
--rag-model-type token --rag-retriever-type search_engine --search_server None \
--dpr-model-file zoo:hallucination/bart_rag_token/model \
--gold-document-titles-key select-docs-titles --insert-gold-docs True \
--beam-min-length 20 --beam-context-block-ngram 3 --beam-block-ngram 3 --beam-block-full-context False --beam-size 10 \
--inference beam --optimizer mem_eff_adam --learningrate 1e-05 --lr-scheduler-patience 1 --model-parallel True \
--batchsize 1 \
--truncate 512 --text-truncate 512 --label-truncate 128 \
--dropout 0.0 --attention-dropout 0.0 \
--min-doc-token-length 64 --max-doc-token-length 256 \
--fp16 True --fp16-impl mem_efficient --force-fp16-tokens True \
--tensorboard-log true --model-file /content/gdrive/MyDrive/BB2AM/model \
--eval-batchsize 8 --skip-generation true\
--vmt ppl -vmm min --metrics all
```
To make the trained model act like BB2 with memory etc. Need to use following command
```bash
python -m parlai interactive --model-file ../BB2AM/model --search_server 0.0.0.0:8080 --loglevel debug --memory-key personas --knowledge-access-method memory_only --skip-generation false --memory_decoder_model_file zoo:blenderbot2/memory_decoder/model # need define memory_decoder_model_file
```

## 07.08.2023
### TODO
- [x] Evaluate BB2 with blended_skill_talk,hotpotqa,coqa
- [x] Evaluate BB2AM with blended_skill_talk,hotpotqa,coqa
- [x] Cherry pick examples for hotpotqa
- [x] Document the validation result
### Validation
The whole process is evaluated by this [notebook](https://colab.research.google.com/drive/1TwSjhEaN4-BIYu0fyf3d7HuBqwNPaHlA). For both BB2 and BB2AM, we evulated each dataset with 1000 examples.
For each model it took 1.5 hours. The results of metrics is in this [sheet](https://docs.google.com/spreadsheets/d/1mdRfmvx0SD7vxq1iN3ihv2WmGQEo_cd98X79h9OMYp8/edit#gid=0), and the cherry picked examples is in this [sheet](https://docs.google.com/spreadsheets/d/1NO14lvn3nTiHY6fTXvmIzSr3fcDplwWDXQ4bcDFilgA/edit#gid=0).
#### Evaluation command
```bash
parlai eval_model -dp data \
--model projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent \
--model-file /content/gdrive/MyDrive/BB2AM/model \
--init-model /content/gdrive/MyDrive/BB2AM/model \
--dict-file /content/gdrive/MyDrive/BB2AM/model.dict \
# for bb2 no need  init-model and dict-file
--task blended_skill_talk,hotpotqa,coqa \
--memory-decoder-model-file "" --knowledge-access-method memory_only --memory-key full_text \
--search-query-generator-model-file zoo:blenderbot2/query_generator/model --search-query-generator-beam-min-length 2 \
--rag-model-type token --rag-retriever-type search_engine --search_server None \
--dpr-model-file zoo:hallucination/bart_rag_token/model \
--skip-generation False \
--batchsize 1 \
--fp16 True \
--dynamic-batching full \
--report-filename /content/gdrive/MyDrive/BB2AM/eva_1000.json \
--world-logs /content/gdrive/MyDrive/BB2AM/world_1000_log.json \
# world log will document each example and it's prediction with metrics
--display-examples False \
--num-examples 1000 \
--log-every-n-secs 120 \
--metrics ppl,f1,accuracy,hits@1,rouge,bleu
```
> **NOTE** For rouge metric, need before running the evaluation first import ntkl
> ```python
> import nltk
> nltk.download('punkt')
> ```

## 08.08.2023
### TODO
- [x] Create new project `bb2am` and find out how to use it
- [x] Create launch.json for debug
### Create new project in parlai
1. Under [projects](./projects/) put your new project
2. To use the new project, need define the `--model` option with the class name of your modle. E.g
```bash
python -m parlai interactive --model projects.bb2am.agents.blenderbot2:BlenderBot2FidAgent \
--model-file zoo:blenderbot2/blenderbot2_400M/model --search_server 0.0.0.0:8080 --loglevel debug
```
### Debug parlai
1. Create launch.json with vscode tool, choose Module (not python)
2. In launch.json need to define `args`, see this [example](.vscode/launch.json)
3. Before debug, make sure already set up the correct intepreter with <kbd>command</kbd>+<kbd>shift</kbd>+<kbd>p</kbd>