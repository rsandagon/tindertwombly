from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from starlette.requests import Request
import logging
import random
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

import tarfile
import tempfile

from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_

class Item(BaseModel):
    raw: str = ""
    personality:str = ""
    personality_raw:List[str] = []
    reply: str = ""
    history: List[str] = []
    temperature: float = 0.7
    top_k:int = 0
    top_p:float = 0.9
    
app = FastAPI()

class ArgsParser:
  def __init__(self, dataset_path, dataset_cache,model,model_checkpoint,max_history,device,no_sample,max_length,min_length,seed,temperature,top_k,top_p):
    self.dataset_path = dataset_path
    self.dataset_cache = dataset_cache
    self.model = model
    self.model_checkpoint = model_checkpoint
    self.max_history = max_history
    self.device = device
    self.no_sample = no_sample
    self.max_length = max_length
    self.min_length = min_length
    self.seed = seed
    self.temperature = temperature
    self.top_k = top_k
    self.top_p = top_p

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

args = ArgsParser('','./dataset_cache','openai-gpt','',2,'cpu','store_true',20,1,0,0.7,0.9,0.9)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
# logger.info(pformat(args))

# tempdir = tempfile.mkdtemp()
# with tarfile.open('./downloads/gpt_personachat_cache.tar.gz', 'r:gz') as archive:
#     archive.extractall(tempdir)
args.model_checkpoint = "./downloads"

if args.seed != 0:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

logger.info("Get pretrained model and tokenizer")
tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)
add_special_tokens_(model, tokenizer)

# save as json file?
# personalities = ["i like to eat Filipino foods .", "i like to go jogging .", "i like to shoot a bow .", "my favorite holiday is Christmas .","my mom is my best friend .", "i have four sisters .", "i believe that mermaids are real .", "i love milk tea ."]
def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)

p1 = ["my wife spends all my money .", "i'm 40 years old .", "i hate my job .", "I work as a car salesman."]
p2 = ["i love listening to britney spears .", "i'm high maintenance' .", "i love spending money .", "i diet a lot ."]
p3 = ["i enjoy gardening and walking outdoors .", "i've a dogs .", "i work as a school teacher .", "i'm a woman ."]

personalities = [p1,p2,p3]
personality_raw = random.choice(personalities)
personality = tokenize(personality_raw)


# origins = [
#     "http:localhost",
#     "http:localhost:5500",
#     "http://localhost:5500/public/index.html",
#     "http://localhost:5500/public/"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.get("/")
async def read_root():
    personality_raw = random.choice(personalities)
    personality = tokenize(personality_raw)
    return {"personality": tokenizer.decode(chain(*personality)),
    "personality_raw":personality_raw
    }

@app.post("/chat")
async def post_chat(request: Request, item: Item):
    if request.method == "POST":
        item_dict = item.dict()
        raw_text = item.raw
        logger.info(">>>  /chat <<<")
        logger.info(item.history)
        history = []
        if not history:
          print("no history yet")
        else:
          for i in item.history: 
            history.append(tokenizer.encode(i))
        history.append(tokenizer.encode(raw_text))
        history = history[-(2*args.max_history+1):]
        args.temperature = item.temperature
        args.top_k = item.top_k
        args.top_p = item.top_p
        # personality
        personality = tokenize(item.personality_raw)
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer, model, args)
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        item_dict.update({"reply":out_text})
        item_dict.update({"personality_raw":item.personality_raw})
        item_dict.update({"personality":tokenizer.decode(chain(*personality))})
    return item_dict

