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
   
## 10.08.2023 - 11.08.2023
### TODO
- [x] Add hybridsummarizer in project `bb2am`
- [x] Try out summarizer with model `philschmid/bart-large-cnn-samsum`
- [x] Try out summarizer with model `philschmid/flan-t5-base-samsum`

### Modify memory decoder
See this [commit](https://github.com/facebookresearch/ParlAI/commit/f14d383f784646b28ff575c0cc9630949a03488d)

### Results
The tryout example see this [doc](https://docs.google.com/document/d/15VTba7NdBMtr73RjLRNOhK85vXCH5nEAYJbJQB-c4m8/edit)

| Model           | With Augmented Memory | Trained            | Can answer "Why did I move to Berlin"? |
| --------------- | --------------------- | ------------------ | -------------------------------------- |
| Trained BB2AM   | :white_check_mark:    | :white_check_mark: | :white_check_mark:                     |
| Trained BB2     | :x:                   | :white_check_mark: | :x:                                    |
| Untrained BB2AM | :white_check_mark:    | :x:                | :x:                                    |
| Untrained BB2   | :x:                   | :white_check_mark: | :x:                                    |

## BB2AM OPT
Total parameters: 732,961,280 (406,286,336 trainable)
12:04:47 | Loading existing model params from ../BB2AM/model
12:04:55 | Opt:
12:04:55 |     activation: gelu
12:04:55 |     adafactor_eps: '[1e-30, 0.001]'
12:04:55 |     adam_eps: 1e-08
12:04:55 |     add_cleaned_reply_to_history: False
12:04:55 |     add_p1_after_newln: False
12:04:55 |     aggregate_micro: False
12:04:55 |     allow_missing_init_opts: False
12:04:55 |     attention_dropout: 0.0
12:04:55 |     batchsize: 1
12:04:55 |     beam_block_full_context: False
12:04:55 |     beam_block_list_filename: None
12:04:55 |     beam_block_ngram: 3
12:04:55 |     beam_context_block_ngram: 3
12:04:55 |     beam_delay: 30
12:04:55 |     beam_length_penalty: 0.65
12:04:55 |     beam_min_length: 20
12:04:55 |     beam_size: 10
12:04:55 |     betas: '[0.9, 0.999]'
12:04:55 |     bpe_add_prefix_space: None
12:04:55 |     bpe_debug: False
12:04:55 |     bpe_dropout: None
12:04:55 |     bpe_merge: None
12:04:55 |     bpe_vocab: None
12:04:55 |     candidates: inline
12:04:55 |     cap_num_predictions: 100
12:04:55 |     checkpoint_activations: False
12:04:55 |     clearml_log: False
12:04:55 |     clearml_project_name: ParlAI
12:04:55 |     clearml_task_name: 'Default Task'
12:04:55 |     codes_attention_num_heads: 4
12:04:55 |     codes_attention_type: basic
12:04:55 |     compressed_indexer_factory: IVF4096_HNSW128,PQ128
12:04:55 |     compressed_indexer_gpu_train: False
12:04:55 |     compressed_indexer_nprobe: 64
12:04:55 |     compute_tokenized_bleu: False
12:04:55 |     data_parallel: False
12:04:55 |     datapath: /Users/su/Desktop/MasterThesis/ParlAI/data
12:04:55 |     datatype: train:stream
12:04:55 |     delimiter: '\n'
12:04:55 |     dict_class: parlai.core.dict:DictionaryAgent
12:04:55 |     dict_endtoken: __end__
12:04:55 |     dict_file: ../BB2AM/model.dict
12:04:55 |     dict_include_test: False
12:04:55 |     dict_include_valid: False
12:04:55 |     dict_initpath: None
12:04:55 |     dict_language: english
12:04:55 |     dict_loaded: True
12:04:55 |     dict_lower: False
12:04:55 |     dict_max_ngram_size: -1
12:04:55 |     dict_maxexs: -1
12:04:55 |     dict_maxtokens: -1
12:04:55 |     dict_minfreq: 0
12:04:55 |     dict_nulltoken: __null__
12:04:55 |     dict_starttoken: __start__
12:04:55 |     dict_textfields: text,labels
12:04:55 |     dict_tokenizer: gpt2
12:04:55 |     dict_unktoken: __unk__
12:04:55 |     display_add_fields: 
12:04:55 |     display_examples: False
12:04:55 |     display_prettify: False
12:04:55 |     doc_chunk_split_mode: word
12:04:55 |     doc_chunks_ranker: head
12:04:55 |     download_path: None
12:04:55 |     dpr_model_file: zoo:hallucination/bart_rag_token/model
12:04:55 |     dpr_num_docs: 25
12:04:55 |     dropout: 0.0
12:04:55 |     dynamic_batching: None
12:04:55 |     embedding_projection: random
12:04:55 |     embedding_size: 1024
12:04:55 |     embedding_type: random
12:04:55 |     embeddings_scale: True
12:04:55 |     encode_candidate_vecs: True
12:04:55 |     encode_candidate_vecs_batchsize: 256
12:04:55 |     eval_batchsize: 8
12:04:55 |     eval_candidates: inline
12:04:55 |     eval_dynamic_batching: None
12:04:55 |     evaltask: None
12:04:55 |     ffn_size: 4096
12:04:55 |     final_extra_opt: 
12:04:55 |     fixed_candidate_vecs: reuse
12:04:55 |     fixed_candidates_path: None
12:04:55 |     force_fp16_tokens: True
12:04:55 |     fp16: True
12:04:55 |     fp16_impl: mem_efficient
12:04:55 |     generation_model: bart
12:04:55 |     gold_document_key: __selected-docs__
12:04:55 |     gold_document_titles_key: select-docs-titles
12:04:55 |     gold_knowledge_passage_key: checked_sentence
12:04:55 |     gold_knowledge_title_key: title
12:04:55 |     gold_sentence_key: __selected-sentences__
12:04:55 |     gpu: -1
12:04:55 |     gpu_beam_blocking: False
12:04:55 |     gradient_clip: 0.1
12:04:55 |     hide_labels: False
12:04:55 |     history_add_global_end_token: None
12:04:55 |     history_reversed: False
12:04:55 |     history_size: -1
12:04:55 |     hnsw_ef_construction: 200
12:04:55 |     hnsw_ef_search: 128
12:04:55 |     hnsw_indexer_store_n: 128
12:04:55 |     ignore_bad_candidates: False
12:04:55 |     image_cropsize: 224
12:04:55 |     image_mode: raw
12:04:55 |     image_size: 256
12:04:55 |     indexer_buffer_size: 65536
12:04:55 |     indexer_type: compressed
12:04:55 |     inference: beam
12:04:55 |     init_fairseq_model: None
12:04:55 |     init_model: data/models/blenderbot2/blenderbot2_400M/model
12:04:55 |     init_opt: None
12:04:55 |     insert_gold_docs: True
12:04:55 |     interactive_candidates: fixed
12:04:55 |     interactive_mode: True
12:04:55 |     interactive_task: True
12:04:55 |     invsqrt_lr_decay_gamma: -1
12:04:55 |     is_debug: False
12:04:55 |     knowledge_access_method: classify
12:04:55 |     label_truncate: 128
12:04:55 |     lambda_decay: 0.9
12:04:55 |     learn_embeddings: True
12:04:55 |     learn_positional_embeddings: True
12:04:55 |     learningrate: 1e-05
12:04:55 |     local_human_candidates_file: None
12:04:55 |     log_every_n_secs: -1
12:04:55 |     log_every_n_steps: 100
12:04:55 |     log_keep_fields: all
12:04:55 |     loglevel: debug
12:04:55 |     lr_scheduler: reduceonplateau
12:04:55 |     lr_scheduler_decay: 0.5
12:04:55 |     lr_scheduler_patience: 1
12:04:55 |     max_doc_token_length: 256
12:04:55 |     max_train_steps: 10000
12:04:55 |     max_train_time: -1
12:04:55 |     memory_attention: sqrt
12:04:55 |     memory_decoder_beam_min_length: 10
12:04:55 |     memory_decoder_beam_size: 3
12:04:55 |     memory_decoder_delimiter: '\n'
12:04:55 |     memory_decoder_ignore_phrase: persona:
12:04:55 |     memory_decoder_key: full_text
12:04:55 |     memory_decoder_model_file: zoo:blenderbot2/memory_decoder/model
12:04:55 |     memory_decoder_one_line_memories: False
12:04:55 |     memory_decoder_truncate: -1
12:04:55 |     memory_doc_delimiter: :
12:04:55 |     memory_doc_title_delimiter: ' / '
12:04:55 |     memory_extractor_phrase: persona:
12:04:55 |     memory_key: personas
12:04:55 |     memory_reader_model: None
12:04:55 |     memory_retriever_truncate: -1
12:04:55 |     memory_writer_model: bert
12:04:55 |     memory_writer_model_file: zoo:hallucination/multiset_dpr/hf_bert_base.cp
12:04:55 |     metrics: all
12:04:55 |     min_doc_token_length: 64
12:04:55 |     model: projects.bb2am.agents.blenderbot2:BlenderBot2FidAgent
12:04:55 |     model_file: ../BB2AM/model
12:04:55 |     model_parallel: True
12:04:55 |     momentum: 0
12:04:55 |     multitask_weights: [1]
12:04:55 |     mutators: None
12:04:55 |     n_decoder_layers: 12
12:04:55 |     n_docs: 5
12:04:55 |     n_encoder_layers: 12
12:04:55 |     n_extra_positions: 0
12:04:55 |     n_heads: 16
12:04:55 |     n_layers: 12
12:04:55 |     n_positions: 1024
12:04:55 |     n_ranked_doc_chunks: 1
12:04:55 |     n_segments: 0
12:04:55 |     nesterov: True
12:04:55 |     no_cuda: False
12:04:55 |     normalize_sent_emb: False
12:04:55 |     num_epochs: 1.0
12:04:55 |     num_workers: 0
12:04:55 |     nus: [0.7]
12:04:55 |     omega_bound: 0.3
12:04:55 |     optimizer: mem_eff_adam
12:04:55 |     outfile: 
12:04:55 |     output_conversion_path: None
12:04:55 |     output_scaling: 1.0
12:04:55 |     override: "{'model': 'projects.bb2am.agents.blenderbot2:BlenderBot2FidAgent', 'model_file': '../BB2AM/model', 'search_server': '0.0.0.0:8080', 'loglevel': 'debug', 'memory_key': 'personas', 'skip_generation': False, 'memory_decoder_model_file': 'zoo:blenderbot2/memory_decoder/model', 'knowledge_access_method': 'classify'}"
12:04:55 |     p_reset: True
12:04:55 |     parlai_home: /content/ParlAI
12:04:55 |     path_to_dense_embeddings: None
12:04:55 |     path_to_dpr_passages: zoo:hallucination/wiki_passages/psgs_w100.tsv
12:04:55 |     path_to_index: zoo:hallucination/wiki_index_compressed/compressed_pq
12:04:55 |     person_tokens: False
12:04:55 |     poly_attention_num_heads: 4
12:04:55 |     poly_attention_type: basic
12:04:55 |     poly_faiss_model_file: None
12:04:55 |     poly_n_codes: 64
12:04:55 |     poly_score_initial_lambda: 0.5
12:04:55 |     polyencoder_init_model: wikito
12:04:55 |     polyencoder_type: codes
12:04:55 |     print_docs: False
12:04:55 |     query_generator_beam_min_length: 2
12:04:55 |     query_generator_beam_size: 1
12:04:55 |     query_generator_delimiter: '\n'
12:04:55 |     query_generator_ignore_phrase: persona:
12:04:55 |     query_generator_inference: beam
12:04:55 |     query_generator_key: full_text
12:04:55 |     query_generator_model_file: zoo:blenderbot2/query_generator/model
12:04:55 |     query_generator_truncate: -1
12:04:55 |     query_model: bert_from_parlai_rag
12:04:55 |     rag_model_type: token
12:04:55 |     rag_query_truncate: 512
12:04:55 |     rag_retriever_query: full_history
12:04:55 |     rag_retriever_type: search_engine
12:04:55 |     rag_turn_discount_factor: 1.0
12:04:55 |     rag_turn_marginalize: doc_then_turn
12:04:55 |     rag_turn_n_turns: 2
12:04:55 |     rank_candidates: False
12:04:55 |     rank_top_k: -1
12:04:55 |     reduction_type: mean
12:04:55 |     regret: False
12:04:55 |     regret_dict_file: None
12:04:55 |     regret_intermediate_maxlen: 32
12:04:55 |     regret_model_file: None
12:04:55 |     regret_override_index: False
12:04:55 |     relu_dropout: 0.0
12:04:55 |     repeat_blocking_heuristic: True
12:04:55 |     retriever_debug_index: None
12:04:55 |     retriever_delimiter: '\n'
12:04:55 |     retriever_embedding_size: 768
12:04:55 |     retriever_ignore_phrase: persona:
12:04:55 |     return_cand_scores: False
12:04:55 |     save_after_valid: False
12:04:55 |     save_every_n_secs: 500.0
12:04:55 |     save_format: conversations
12:04:55 |     search_query_generator_beam_min_length: 2
12:04:55 |     search_query_generator_beam_size: 1
12:04:55 |     search_query_generator_inference: greedy
12:04:55 |     search_query_generator_model_file: zoo:blenderbot2/query_generator/model
12:04:55 |     search_query_generator_text_truncate: 512
12:04:55 |     search_server: 0.0.0.0:8080
12:04:55 |     seed: None
12:04:55 |     share_encoders: True
12:04:55 |     share_search_and_memory_query_encoder: False
12:04:55 |     share_word_embeddings: True
12:04:55 |     short_final_eval: False
12:04:55 |     single_turn: False
12:04:55 |     skip_generation: False
12:04:55 |     skip_retrieval_token: no_passages_used
12:04:55 |     skip_search_key: skip_search
12:04:55 |     special_tok_lst: None
12:04:55 |     split_lines: True
12:04:55 |     splitted_chunk_length: 256
12:04:55 |     starttime: Aug04_14-25
12:04:55 |     t5_dropout: 0.0
12:04:55 |     t5_generation_config: None
12:04:55 |     t5_model_arch: t5-base
12:04:55 |     t5_model_parallel: False
12:04:55 |     task: blended_skill_talk,hotpotqa,coqa
12:04:55 |     teacher_seed: None
12:04:55 |     temperature: 1.0
12:04:55 |     tensorboard_log: True
12:04:55 |     tensorboard_logdir: None
12:04:55 |     text_truncate: 512
12:04:55 |     tfidf_max_doc_paragraphs: -1
12:04:55 |     tfidf_model_path: zoo:wikipedia_full/tfidf_retriever/model
12:04:55 |     thorough: False
12:04:55 |     topk: 10
12:04:55 |     topp: 0.9
12:04:55 |     train_predict: False
12:04:55 |     truncate: 512
12:04:55 |     update_freq: 1
12:04:55 |     use_memories: False
12:04:55 |     use_reply: label
12:04:55 |     validation_cutoff: 1.0
12:04:55 |     validation_every_n_epochs: -1
12:04:55 |     validation_every_n_secs: -1
12:04:55 |     validation_every_n_steps: 1000
12:04:55 |     validation_max_exs: -1
12:04:55 |     validation_metric: ppl
12:04:55 |     validation_metric_mode: min
12:04:55 |     validation_patience: 5
12:04:55 |     validation_share_agent: False
12:04:55 |     variant: prelayernorm
12:04:55 |     verbose: False
12:04:55 |     wandb_entity: None
12:04:55 |     wandb_log: False
12:04:55 |     wandb_log_model: False
12:04:55 |     wandb_name: None
12:04:55 |     wandb_project: None
12:04:55 |     warmup_rate: 0.0001
12:04:55 |     warmup_updates: -1
12:04:55 |     weight_decay: None
12:04:55 |     woi_doc_chunk_size: 500
12:04:55 |     world_logs: 
12:04:55 |     wrap_memory_encoder: False