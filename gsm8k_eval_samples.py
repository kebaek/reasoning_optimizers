from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import argparse
import re 
from utils import is_equiv
import json

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--eval_type", type=str, default="test")
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--temp", type=float, default=0.8)


args = parser.parse_args()


llm = LLM(model=args.ckpt_dir, tensor_parallel_size=1, trust_remote_code=True)  # Name or path of your model


if args.eval_type == "test":
    dataset = load_dataset("gsm8k", "main")

    test_questions = dataset["test"]["question"]
    test_answers = dataset["test"]['answer']
    eval_questions = test_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = test_answers
elif args.eval_type == "test_small":
    dataset = load_dataset("gsm8k", "main")

    test_questions = dataset["test"]["question"][:10]
    test_answers = dataset["test"]['answer'][:10]
    eval_questions = test_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = test_answers
elif args.eval_type == "train":
    dataset = load_dataset("gsm8k", "main")
    train_questions = dataset["train"]["question"]
    train_answers = dataset["train"]['answer']
    eval_questions = train_questions
    eval_questions = [question + "\nAnswer:" for question in train_questions]
    eval_answers = train_answers
elif args.eval_type == "train_first":
    dataset = load_dataset("gsm8k", "main")
    train_questions = dataset["train"]["question"]
    train_answers = dataset["train"]['answer']
    eval_questions = [question + "\nAnswer: First" for question in train_questions]
    eval_answers = train_answers
elif args.eval_type == "train_we":
    dataset = load_dataset("gsm8k", "main")
    train_questions = dataset["train"]["question"]
    train_answers = dataset["train"]['answer']
    eval_questions = [question + "\nAnswer: We know that" for question in train_questions]
    eval_answers = train_answers
    

def get_aug_answer(full_answer):
    idx = full_answer.rfind("The answer is")
    if idx == -1:
        return None
    else:
        answer = full_answer[idx + len("The answer is: "):]
        answer = answer.replace(":", "").replace("$", "").strip()
        if len(answer)> 0:
            if answer[-1] == ".":
                answer = answer[:-1]
            left = "\\boxed{"
            if answer[:len(left)] == left and answer[-1] == "}":
                answer = answer[len(left):-1]
        return answer.replace(",", "")

def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    return text[start:].replace(",", "")

def answer_type_individial(output , answer):
    answer = extract_latex(answer)

    output_answer = get_aug_answer(output)
    if output_answer == None:
        output_answer = extract_latex(output)
    if output_answer is not None and answer is not None:
        eqiv = is_equiv(answer, output_answer, verbose=False)

        if eqiv:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


sampling_params = SamplingParams(
    n = args.num_samples,
    temperature=args.temp,
    max_tokens=512,
    top_p=0.95,
    seed=args.seed,
    stop="\nDone."
)

output = llm.generate(eval_questions, sampling_params)

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


np.save(os.path.join(args.ckpt_dir, f"{args.eval_type}_answers{args.num_samples}_seed{args.seed}_temp{args.temp}.npy"), answers_all)
np.save(os.path.join(args.ckpt_dir, f"{args.eval_type}_answer_types{args.num_samples}_seed{args.seed}_temp{args.temp}.npy"), answer_types_all)