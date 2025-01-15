import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import normalize
from datasets import load_dataset
import numpy as np
import os
from utils import *
from tqdm import tqdm 
import matplotlib.pyplot as plt

# The path of your model after cloning it
model_dir = '/home/kbaek/.cache/huggingface/hub/models--dunzhang--stella_en_400M_v5/snapshots/db4ace10eb6a7131d349077b2eccc5c76a77277b/'

vector_dim = 1024
vector_linear_directory = f"2_Dense_{vector_dim}"
text_encoder = AutoModel.from_pretrained("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda().eval()
# you can also use this model without the features of `use_memory_efficient_attention` and `unpad_inputs`. It can be worked in CPU.
# model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,use_memory_efficient_attention=False,unpad_inputs=False).cuda().eval()
text_encoder_tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_400M_v5", trust_remote_code=True)
# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="dunzhang/stella_en_400M_v5", filename='2_Dense_1024/pytorch_model.bin') 
vector_linear = torch.nn.Linear(in_features=text_encoder.config.hidden_size, out_features=vector_dim)
vector_linear_dict = {
    k.replace("linear.", ""): v for k, v in
    torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
}
vector_linear.load_state_dict(vector_linear_dict)
vector_linear.cuda()

def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    if "\nDone." in text:
        end = text.find("\nDone.")
    elif "<|end_of_text|>" in text: 
        end = text.find("<|end_of_text|>")
    else:
        end = len(text)
    return text[start:end].replace(",", "")

def answer_type_individial(output , answer):
    answer = extract_latex(answer)
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

def evaluate(preds, test_answers, return_all_types=False):
    answer_types_all = [] 
    answer_pred = zip(test_answers, preds)
    for a, ps in answer_pred:
        answer_types_all.append([answer_type_individial(a, p) for p in ps])

    answer_types_all = np.array(answer_types_all)
    print('Right', (answer_types_all==0).mean(axis=-1).mean()) #ratio correct
    print('Wrong', (answer_types_all==1).mean(axis=-1).mean()) #ratio incorrect
    print('Weird', (answer_types_all==2).mean(axis=-1).mean()) #ratio weird formatting

    if return_all_types:
        return (answer_types_all==0).mean(axis=-1).mean(), answer_types_all
    else:
        return (answer_types_all==0).mean(axis=-1).mean()
    
from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
      super().__init__()
      self.stops = stops
      self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      stop_count = 0
      for stop in self.stops:
        # stop_count = (input_ids[0] == stop[0]).sum().item()
        stop_count = int(input_ids[0][-1] == stop[0])

      if stop_count >= self.ENCOUNTERS:
          return True
      return False
    
def process_trace(trace):
    trace = trace.split('Done.')[0]
    trace = trace.split('Answer:')[1]
    return trace 

def default_sampling(tokens, model, tokenizer, temperature=1.0, max_tokens=350, top_p=0.95, num_return_sequences=4):
  stop="\nDone."
  bad_words_ids = [tokenizer(stop).input_ids[1:]]

  predictions = model.generate(**tokens, do_sample=True, temperature=temperature, max_new_tokens=max_tokens, top_p=top_p, bad_words_ids=bad_words_ids, num_return_sequences=num_return_sequences)
  samples = [process_trace(tokenizer.decode(o)) for o in predictions]

  return samples

def concept_sampling(tokens, model, tokenizer, temperature=1.0, max_tokens=350, top_p=0.95, num_return_sequences=4, number_of_sampled_trajs=16):
  stop="\n"
  bad_words_ids = [tokenizer(stop).input_ids[2:]]#, tokenizer("\nDone.").input_ids[2:]]
  print(bad_words_ids)
  stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = bad_words_ids)])
  DONE = False
  inputs = [tokens]
  while not DONE:
    n = number_of_sampled_trajs // len(inputs)
    predictions = []
    for x in inputs:
      for _ in tqdm(range(n)):
        p = model.generate(**x, do_sample=True, temperature=temperature, max_new_tokens=max_tokens, top_p=top_p, stopping_criteria=stopping_criteria)
        predictions.append(p[0])
    samples = [tokenizer.decode(o) for o in predictions]
    with torch.no_grad():
      input_data = text_encoder_tokenizer([process_trace(s) for s in samples], padding="longest", truncation=True, max_length=512, return_tensors="pt")
      input_data = {k: v.cuda() for k, v in input_data.items()}
      attention_mask = input_data["attention_mask"]
      last_hidden_state = text_encoder(**input_data)[0]
      last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
      docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
      docs_vectors = normalize(vector_linear(docs_vectors))
      similarities = docs_vectors @ docs_vectors.T 

    # choose 4 most disimiliar samples using a greedy strategy. start with random sample, then choose most disimilar to it, and so on.
    chosen_samples = []
    chosen_samples.append(torch.randint(0, number_of_sampled_trajs, (1,)).item())
    for _ in range(num_return_sequences - 1):
      disimilarities = similarities[chosen_samples[-1]]
      disimilarities[chosen_samples] = 2
      chosen_samples.append(disimilarities.argmin().item())
    inputs = [tokenizer(samples[i], return_tensors='pt').to('cuda:1') for i in chosen_samples]
    print(samples[0])
    DONE = np.sum([int('####' in samples[i]) for i in chosen_samples]) == num_return_sequences
    print(DONE)
    
  return [process_trace(samples[i]) for i in chosen_samples]

dataset = load_dataset("gsm8k", "main")
test_questions = dataset["test"]["question"]
test_answers = dataset["test"]['answer']
eval_questions = test_questions
eval_questions = [question + "\nAnswer:" for question in eval_questions]
eval_answers = test_answers
np.random.seed(0)
random_100_idxs = np.random.choice(len(eval_questions), 40, replace=False)
eval_questions = [eval_questions[i] for i in random_100_idxs]
eval_answers = [eval_answers[i] for i in random_100_idxs]


ckpt_dir = '/data/locus/llm_weights/kbaek/ckpts/gsm8k_orig_10epochs_full_lr2e-05_rho0_bs128_mistral-7b-v0.1_WANDBm6j3wdw9/checkpoint-580/'
use_auth_token = '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'
cache_dir = '/tmp/kbaek/hf_cache'

tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, 
                                          cache_dir=cache_dir,
                                          use_auth_token = use_auth_token,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt_dir, 
                                            cache_dir=cache_dir,
                                            use_auth_token = use_auth_token,
                                            trust_remote_code=True)

special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=model,
)

model.generation_config.pad_token_id = tokenizer.pad_token_id
model.to("cuda:1")
model.eval()

preds = []
for x in tqdm(eval_questions):
  tokens = tokenizer(x, return_tensors='pt').to('cuda:1')
  preds.append(concept_sampling(tokens, model, tokenizer, temperature=1.0, number_of_sampled_trajs=8))
acc, answer_types_all = evaluate(preds, eval_answers, True)

answer_types_all, preds = np.array(answer_types_all), np.array(preds)
np.save('concept_answer_types_all.npy', answer_types_all)
np.save('concept_preds.npy', preds) 
print(np.mean([0 in a for a in answer_types_all]))

