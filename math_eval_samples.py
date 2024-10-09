from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import argparse
import re 
import json
from utils import is_equiv

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--eval_type", type=str, default="test")
parser.add_argument("--num_devices", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=5)

args = parser.parse_args()

dataset = load_dataset("hendrycks/competition_math")
train_questions = np.array(dataset["train"]["problem"])
train_answers = np.array(dataset["train"]['solution'])

test_questions = np.array(dataset["test"]["problem"])
test_answers = np.array(dataset["test"]['solution'])

sampling_params = SamplingParams(
    n = args.num_samples,
    temperature=0.8,
    max_tokens=1024,
    seed=args.seed,
    stop="\nDone."
)
    

if args.eval_type == "test":
    eval_questions = test_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = test_answers
elif args.eval_type == "train":
    eval_questions = train_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = train_answers
elif args.eval_type == "train_first":
    train_questions = train_questions
    train_answers = train_answers
    eval_questions = [question + "\nAnswer: First" for question in train_questions]
    eval_answers = train_answers
elif args.eval_type == "train_we":    
    train_questions = train_questions
    train_answers = train_answers
    eval_questions = [question + "\nAnswer: We know that" for question in train_questions]
    eval_answers = train_answers

llm = LLM(model=args.ckpt_dir, tensor_parallel_size=args.num_devices)  # Name or path of your model
output = llm.generate(eval_questions, sampling_params)



def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def get_aug_answer(full_answer):
    answer_phrases = ["FINAL ANSWER", "answer is", "ANSWER", "Final Answer", "is:", "Answer", "answer"]
    
    for answer in answer_phrases:
        idx = full_answer.rfind(answer)
        if idx != -1:
            answer = full_answer[idx + len(answer):]
            answer = answer.replace(":", "").replace("$", "").replace("\n", "").strip()
            if len(answer)> 0:
                if answer[-1] == ".":
                    answer = answer[:-1]
                left = "\\boxed{"
                if answer[:len(left)] == left and answer[-1] == "}":
                    answer = answer[len(left):-1]
            return answer 
    return None

def answer_type_individial(output , answer):
    answer = remove_boxed(last_boxed_only_string(answer))
    output_answer = remove_boxed(last_boxed_only_string(output))
    if output_answer == None:
        output_answer = get_aug_answer(output)

    if output_answer is not None:
        
        eqiv = is_equiv(answer, output_answer, verbose=False)

        if eqiv:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


answer_types_all = []
answers_all = []
for i in range(len(output)):
    answer_types = []
    answers = []
    for item in output[i].outputs:
        answers.append(item.text)
        answer_type = answer_type_individial(item.text, eval_answers[i])
        answer_types.append(answer_type)
    answer_types_all.append(answer_types)
    answers_all.append(answers)

answer_types_all = np.array(answer_types_all)
answers_all = np.array(answers_all)
print((answer_types_all==0).mean(axis=-1).mean()) #ratio correct
print((answer_types_all==1).mean(axis=-1).mean()) #ratio incorrect
print((answer_types_all==2).mean(axis=-1).mean()) #ratio weird formatting

np.save(os.path.join(args.ckpt_dir, f"{args.eval_type}_answers{args.num_samples}_seed{args.seed}.npy"), answers_all)
np.save(os.path.join(args.ckpt_dir, f"{args.eval_type}_answer_types{args.num_samples}_seed{args.seed}.npy"), answer_types_all)