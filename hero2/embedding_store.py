import os
import json
import pickle
import tqdm
import torch
import multiprocessing
from sentence_transformers import SentenceTransformer

MODEL_NAME    = 'Alibaba-NLP/gte-base-en-v1.5'
KNOWLEDGE_DIR = "knowledge_store/dev"
OUTPUT_DIR    = "gte-base-large-v1.5-embed"
NUM_FILES     = 500
BATCH_SIZE    = 32

model = None

def init_worker(device_queue):
    global model
    if torch.cuda.is_available():
        device_id = device_queue.get()
        device = f"cuda:{device_id}"
    else:
        device = "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"[Worker PID {os.getpid()}] loaded model on {device}")

def process_file(i):
    input_path = os.path.join(KNOWLEDGE_DIR, f"{i}.json")
    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if len(data['url2text']) > 0:
                docs.append("\n".join(data["url2text"]))
    if not docs:
        return i, 0

    embeddings = model.encode(docs, batch_size=BATCH_SIZE, show_progress_bar=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{i}.pkl")
    with open(out_path, "wb") as out_f:
        pickle.dump(embeddings, out_f)

    return i, len(docs)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    
    if torch.cuda.is_available():
        ngpu = max(1, torch.cuda.device_count())
    else:
        ngpu = 1
    devices_queue = multiprocessing.Queue()
    for gpu_id in range(ngpu):
        devices_queue.put(gpu_id)

    # Pool 생성 & 병렬 처리
    with multiprocessing.Pool(
        processes=ngpu,
        initializer=init_worker,
        initargs=(devices_queue,)
    ) as pool:
        # tqdm으로 진행상황 보기
        for idx, doc_count in tqdm.tqdm(
            pool.imap_unordered(process_file, range(NUM_FILES)),
            total=NUM_FILES,
            desc="Embedding files"
        ):
            if doc_count == 0:
                tqdm.tqdm.write(f"[{idx}] skipped, no docs")
            else:
                tqdm.tqdm.write(f"[{idx}] embedded {doc_count} docs")