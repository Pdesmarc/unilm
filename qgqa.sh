#!/bin/bash

DATA_DIR=qa_data

# run decoding
python src/qa/run_QAinference.py \
  --input_para_ans ${DATA_DIR}/geologieTexte2.txt \
  --input_questions ${DATA_DIR}/geologie_questions2.txt \
  --output_file ${DATA_DIR}/geologie_QA.txt 

