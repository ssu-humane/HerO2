import argparse
import copy
import json
import tqdm
import os
from vllm import LLM, SamplingParams
import torch

prompt = """Your task is to read the following document carefully and summarize it into a single, coherent paragraph. Focus on capturing the main ideas and essential details without adding new information or personal opinions.
Document:
{}
"""

def all_documents(knowledge_file, top_k=-1):
  documents, urls = [], []
  
  with open(knowledge_file, "r", encoding="utf-8") as json_file:
    for i, line in enumerate(json_file):
        if len(urls) == top_k: break
      
        data = json.loads(line)
        if len(data["url2text"]) > 0:
            documents.append(data["url2text"])
            urls.append(data['url'])
  return documents, urls

def truncate_chat_prompt(prompt: str, tokenizer, max_len: int) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
        return tokenizer.decode(token_ids, add_special_tokens=False)
    else:
        return prompt

def main(args):
    gpu_counts = torch.cuda.device_count()
    print(f"Using {gpu_counts} GPU{'s' if gpu_counts > 1 else ''}")
    
    llm = LLM(
        model= args.model,
        tensor_parallel_size=gpu_counts, 
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams( # https://huggingface.co/Qwen/Qwen3-8B  제안된 하이퍼파라미터
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        skip_special_tokens=False,
        max_tokens=4096,
    )
    
    for i in tqdm.tqdm(range(500)):
        documents, urls = all_documents(f"{args.knowledge_store}/{i}.json")
    
        batch_prompt = []
        for pd in documents:
            target_prompt = copy.deepcopy(prompt)
            truncate_doc = truncate_chat_prompt("\n".join(pd), tokenizer, max_len=args.max_input_token_size)
            target_prompt = target_prompt.format(truncate_doc)
            target_prompt = llm.get_tokenizer().apply_chat_template([{"role": "user", "content": target_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            batch_prompt.append(target_prompt)
        
        responses = llm.generate(batch_prompt, sampling_params, use_tqdm=False)
        results = [response.outputs[0].text.strip() for response in responses]

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        with open(f"{args.save_path}/{i}.json", "w", encoding="utf-8") as f:
            for er, url in zip(results, urls):
                json_line = {
                    "url2text": er,
                    "url": url
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_store', default='knowledge_store/dev')
    parser.add_argument('--save_path', default='knowledge_store/dev_summary')
    parser.add_argument('--max_input_token_size', type=int ,default=24000)
    parser.add_argument('-m', '--model', default="Qwen/Qwen3-8B")
    args = parser.parse_args()
    main(args)
  
  