import os
import json
import pickle
import tqdm
import torch
import argparse
import multiprocessing
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Alibaba-NLP/gte-base-en-v1.5")
    parser.add_argument("--knowledge_dir", type=str, default="knowledge_store/dev")
    parser.add_argument("--output_dir", type=str, default="knowledge_store/dev_embed")
    parser.add_argument("--num_files", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

model = None

def init_worker(device_queue, model_name):
    global model
    if torch.cuda.is_available():
        device_id = device_queue.get()
        device = f"cuda:{device_id}"
    else:
        device = "cpu"
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    print(f"[Worker PID {os.getpid()}] loaded model on {device}")

def process_file(args):
    i, knowledge_dir, output_dir, batch_size = args
    input_path = os.path.join(knowledge_dir, f"{i}.json")
    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if len(data['url2text']) > 0:
                docs.append("\n".join(data["url2text"]))
    if not docs:
        return i, 0

    embeddings = model.encode(docs, batch_size=batch_size, show_progress_bar=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{i}.pkl")
    with open(out_path, "wb") as out_f:
        pickle.dump(embeddings, out_f)

    return i, len(docs)

if __name__ == "__main__":
    args = parse_args()

    multiprocessing.set_start_method("spawn", force=True)

    if torch.cuda.is_available():
        ngpu = max(1, torch.cuda.device_count())
    else:
        ngpu = 1
    devices_queue = multiprocessing.Queue()
    for gpu_id in range(ngpu):
        devices_queue.put(gpu_id)

    with multiprocessing.Pool(
        processes=ngpu,
        initializer=init_worker,
        initargs=(devices_queue, args.model_name)
    ) as pool:
        task_args = [(i, args.knowledge_dir, args.output_dir, args.batch_size) for i in range(args.num_files)]

        for idx, doc_count in tqdm.tqdm(
            pool.imap_unordered(process_file, task_args),
            total=args.num_files,
            desc="Embedding files"
        ):
            if doc_count == 0:
                tqdm.tqdm.write(f"[{idx}] skipped, no docs")
            else:
                tqdm.tqdm.write(f"[{idx}] embedded {doc_count} docs")