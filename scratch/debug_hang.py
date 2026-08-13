import threading
import sys
import time

def dump_threads():
    while True:
        time.sleep(30)
        print("\n--- DUMPING THREADS ---")
        for thread_id, frame in sys._current_frames().items():
            print(f"Thread {thread_id}:")
            import traceback
            traceback.print_stack(frame)
        print("-----------------------\n")

threading.Thread(target=dump_threads, daemon=True).start()

import os
os.environ["PYTHONPATH"] = "src"
import runpy
sys.argv = ["benchmarks/legalbenchrag/run_benchmark_mini_docname.py", "-rc", "configs/standard_rag.json"]
runpy.run_path("benchmarks/legalbenchrag/run_benchmark_mini_docname.py", run_name="__main__")
