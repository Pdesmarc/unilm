import argparse
import torch
import transformers
from transformers import *
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_para_ans", default=None, type=str,
                        help="The file which contains UNILM QG Paragraphs and Answers.")
    parser.add_argument("--input_questions", default=None, type=str,
                        help="The file which contains UNILM QG pred questions.")
    parser.add_argument("--output_file", default=None, type=str,
                        help="The file to store the output of the BERT QA inference.")
    args = parser.parse_args()
    
    
    if args.input_para_ans:
        with open(args.input_para_ans, encoding="utf-8") as fip:
            input_p = [x.strip().split('[SEP]')[0] for x in fip.readlines()]
            #print("\n {} \n".format(input_p[0]))
        
        
        with open(args.input_para_ans, encoding="utf-8") as fir: 
            input_r = [x.strip().split('[SEP]')[1] for x in fir.readlines()]
            #print("\n {} \n".format(input_r[0]))
    
    if args.input_questions:
        with open(args.input_questions, encoding="utf-8") as fiq:
            input_q = [x.strip() for x in fiq.readlines()]
            #print(input_q[0])

    #print(len(input_q))
    #print(len(input_p))
    output_lines = []
    
    # SQuAD 1.1
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    for j in range(len(input_q)):
        input_ids = tokenizer.encode(input_q[j], input_p[j])
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        answer = answer.replace(' ##', '')
        print(answer)
        output_lines.append(answer)
        #output_lines



    if args.output_file:
        fn_out = args.output_file
        with open(fn_out, "w", encoding="utf-8") as fout:
            for l in output_lines:
                fout.write(l)
                fout.write("\n")

if __name__ == "__main__":
    main()
