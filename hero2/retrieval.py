import json
import pickle
import numpy as np
import tqdm
import argparse
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from datetime import datetime, timedelta
import time

def format_time(seconds: float) -> str:
    """Format time duration nicely."""
    return str(timedelta(seconds=round(seconds)))

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

def main(args):
    script_start = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(args.target_data, "r", encoding='utf-8') as f:
        target_data = json.load(f)
  
    model = SentenceTransformer(args.model, trust_remote_code=True)
  
    with open(args.json_output, "w", encoding="utf-8") as json_output:
        for idx, example in tqdm.tqdm(enumerate(target_data)):
            claim = example['claim']
            query = [claim] + [le for le in example['hypo_fc_docs']  if len(le.strip()) > 0 ]
            query_embeddings = model.encode(query)
            avg_emb_q = np.mean(query_embeddings, axis=0)
            hyde_vector = avg_emb_q.reshape((1, -1))
      
            with open(f"{args.embedding_store_dir}/{example['claim_id']}.pkl", "rb") as f:
                doc_embedding = pickle.load(f)
            
            scores = cos_sim(hyde_vector, doc_embedding)
            scores = np.asarray(scores)
            top_k_idx = np.argsort(scores)[0][::-1][:args.top_k]
            
            documents, urls = all_documents(f"{args.knowledge_store_dir}/{example['claim_id']}.json")
                
            top_k_documents, top_k_urls = [documents[tki] for tki in top_k_idx], [urls[tki] for tki in top_k_idx]

            json_line = {
                "claim_id": example['claim_id'],
                "claim": claim,
                f"top_{args.top_k}": [{"sentence": sent, "url": url} for sent, url in zip(top_k_documents, top_k_urls)]
            }

            json_output.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            json_output.flush()


    total_time = time.time() - script_start
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\nTiming Summary:")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total runtime: {format_time(total_time)}")
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Alibaba-NLP/gte-base-en-v1.5')
    parser.add_argument('--target_data', default='data_store/shero/hyde_fc.json')
    parser.add_argument('--embedding_store_dir', default='embedding_store/test_2025')
    parser.add_argument('--knowledge_store_dir', default='knowledge_store/test_2025')
    parser.add_argument('--json_output', default="data_store/shero/dev_summarized.json")
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()
    main(args)