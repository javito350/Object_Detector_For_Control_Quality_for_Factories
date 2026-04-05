import multiprocessing as mp
import time
import numpy as np
import pandas as pd
from models.memory_bank import MemoryBank

def worker(bank, query_data, iterations, pipe):
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = bank.query(query_data)
        latencies.append((time.perf_counter() - start) * 1000) # ms
    pipe.send(latencies)

def run_stress_test(num_streams):
    # Mock data for query
    query = np.random.randn(1, 1024).astype(np.float32)
    bank = MemoryBank(dimension=1024, use_pq=True)
    bank.build([np.random.randn(1000, 1024).astype(np.float32)])

    processes = []
    pipes = []
    
    for i in range(num_streams):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker, args=(bank, query, 100, child_conn))
        p.start()
        processes.append(p)
        pipes.append(parent_conn)

    all_latencies = []
    for conn in pipes:
        all_latencies.extend(conn.recv())
    
    for p in processes: p.join()
    
    return np.percentile(all_latencies, 99) # Return P99 Latency

if __name__ == "__main__":
    stream_counts = [1, 8, 16, 32, 64]
    results = [{"streams": s, "p99_latency": run_stress_test(s)} for s in stream_counts]
    pd.DataFrame(results).to_csv("./presentation_results/concurrency_scaling.csv")
    print("Concurrency Scaling Data Generated.")