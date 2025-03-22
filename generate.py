import torch
torch.use_deterministic_algorithms(True)
import sys
from models.TransformerLM import *
from data.TinyStories import *
from spacy.tokenizer import Tokenizer
torch.serialization.add_safe_globals([Vocabulary, Tokenizer])
from torch.distributions import Categorical
torch.backends.cudnn.deterministic = True

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

MAX_LENGTH = 500

def main():
   CHKPT_PATH = "./chkpts/GHylu1_TinyStories"
   chkpt = torch.load(CHKPT_PATH, weights_only=False)
   config = chkpt['config']

   print(CHKPT_PATH +" // "+str(chkpt['epoch']))
   # load vocab
   vocab = chkpt['vocab']

   # load model
   model = TransformerLM(len(vocab), config["d_model"], config["n_heads"], config["n_layers"])
   model.load_state_dict(chkpt['model_state_dict'])
   model.to(device)
   
   while True:
    # ask for prompt
    prompt = input("\n\nPrompt:\n")

    # numeralize prompt
    num_prompt = vocab.text2idx(prompt)
    l = len(num_prompt)

    
    for sampler in [argmaxDecode, sampleDecode, nucleusDecode]:
      torch.manual_seed(0)
      random.seed(0)
      torch.cuda.manual_seed(0)
      torch.cuda.manual_seed_all(0)
      
      src = torch.zeros(1,MAX_LENGTH)
      src[0,0] = 1 # <SOS>
      src[0,1:l+1] = torch.Tensor(num_prompt)
      src = src.to(dtype=int, device=device)
      print("\n\n")
      print(sampler)
      print(prompt, end="",flush=True)
      for t in range(l+1,MAX_LENGTH):
          out = model(src)

          src[0,t] =  sampler(out[:,t-1,:])
          
          w = vocab.idx2text([src[0,t].cpu().item()])[0]

          if w == "<EOS>":
              break
          if not any(x in w for x in [".",",","\"","'","!","?"]):
              w = " "+w
          
          print(w,  end='',flush=True)
      print("\n")
   sys.exit(1)


def argmaxDecode(scores):
   return torch.argmax(scores, dim=1)

def sampleDecode(scores, temp = 0.5):
   probs = nn.functional.softmax(scores/temp, dim=1)
   return Categorical(probs).sample()

def nucleusDecode(scores, p=0.9, temp = 0.5):
   probs = nn.functional.softmax(scores/temp, dim=1)
   sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
   cumulative_probs = torch.cumsum(sorted_probs, dim=1)
   k = torch.sum(cumulative_probs < p, dim=1)
   new_probs = torch.zeros_like(probs)
   for i in range(probs.size(0)):
      batch_k = sorted_indices[i,:k[i]+1]
      new_probs[i,batch_k] = probs[i,batch_k]
   
  
   new_probs /= new_probs.sum(dim=1, keepdim=True)
   return Categorical(new_probs).sample()
   

main()