#!/bin/bash

DATA_DIR=DONNEES/test
MODEL_RECOVER_PATH=MODELE/qg_model.bin
EVAL_SPLIT=test
export PYTORCH_PRETRAINED_BERT_CACHE=tmp/bert-cased-pretrained-cache

# run decoding
python src/biunilm/decode_seq2seq.py --bert_model bert-large-cased --new_segment_ids --mode s2s \
  --input_file ${DATA_DIR}/divers.txt --split ${EVAL_SPLIT} \
  --output_file ${DATA_DIR}/new2_questions.txt \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_tgt_length 48 \
  --batch_size 1 --beam_size 1 --length_penalty 0

# --tokenized_input = Whether the text is tokenized with WordPiece
# run evaluation using our tokenized data as reference
python src/qg/eval_on_unilm_tokenized_ref.py --out_file src/qg/output/qg.test.output.txt
# run evaluation using tokenized data of Du et al. (2017) as reference
#python src/qg/eval.py --out_file src/qg/output/qg.test.output.txt
