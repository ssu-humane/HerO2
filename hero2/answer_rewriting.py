import copy
import json
import tqdm
from vllm import LLM, SamplingParams
import argparse
import torch
from datetime import datetime, timedelta
import time


prompt = """The following text provides an evidence obtained through web searches related to a specific question, used for verifying the accuracy of a claim.
Your task is to answer the question based on this evidence. Ensure your answer is Supported by relevant context from the evidence.

Claim: {}
Question: {}

Evidence: {}
"""

def format_time(seconds: float) -> str:
    """Format time duration nicely."""
    return str(timedelta(seconds=round(seconds)))

# def truncate_chat_prompt(prompt: str, tokenizer, max_len: int) -> str:
#     token_ids = tokenizer.encode(prompt, add_special_tokens=False)
#     if len(token_ids) > max_len:
#         token_ids = token_ids[:max_len]
#         return tokenizer.decode(token_ids, add_special_tokens=False)
#     else:
#         return prompt

def main(args):
    script_start = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gpu_counts = torch.cuda.device_count()
    print(f"Using {gpu_counts} GPU{'s' if gpu_counts > 1 else ''}")
  
    llm = LLM(
        model= args.model,
        tensor_parallel_size=gpu_counts, 
        max_model_len=8192,
        gpu_memory_utilization=0.95,
        dtype=torch.bfloat16,
        enforce_eager=True,
        trust_remote_code=True,
    )
  
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        skip_special_tokens=False,
        max_tokens=512,
    )

    data = []
    with open(args.target_data, "r", encoding="utf-8") as f:
        for line in f:
          data.append(json.loads(line))

    with open(args.json_output, "w", encoding="utf-8") as output_json:
        for d in tqdm.tqdm(data):
            all_prompts = []
            for item in d["evidence"]:
                target_prompt = copy.deepcopy(prompt)
                target_prompt = target_prompt.format(d['claim'], item['question'], item['answer'])
                target_prompt = llm.get_tokenizer().apply_chat_template([{"role": "user", "content": target_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                # target_prompt = truncate_chat_prompt(target_prompt, llm.get_tokenizer(), max_len=3584)
                target_prompt += "Answer: "
                all_prompts.append(target_prompt)
        
            responses = llm.generate(all_prompts, sampling_params, use_tqdm=False)
            results = [response.outputs[0].text.strip() for response in responses]
      
            new_evidence = []
            for num, item in enumerate(d["evidence"]):
                if results[num].lower() != "none":
                    item['answer'] = results[num]
                    new_evidence.append(item)
            d['evidence'] = new_evidence

            output_json.write(json.dumps(d, ensure_ascii=False) + "\n")
            output_json.flush()

    total_time = time.time() - script_start
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\nTiming Summary:")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total runtime: {format_time(total_time)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--target_data', default='data_store/shero/dev_top_k_qa.json') # qa pair
    parser.add_argument('-o', '--json_output', default='data_store/shero/dev_top_k_qa_rewrite.json')
    parser.add_argument('-m', '--model', default="Qwen/Qwen3-8B")
    args = parser.parse_args()
    main(args)