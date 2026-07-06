import time
import os
import psutil

try:
    import resource
except ImportError:
    resource = None

def run_graphlens_with_profiling():
    process = psutil.Process(os.getpid())
    
    print("Initializing environment isolation...")
    # Trigger a quick initial footprint baseline
    baseline_mem_bytes = process.memory_info().rss
    start_time = time.perf_counter()
    
    # =========================================================================
    # TARGET YOUR ACTUAL GRAPHLENS SCRIPT HERE
    # =========================================================================
    target_script = "graphlens.py"  # <-- Change this to your actual main pipeline file name!
    
    print(f"GraphLens Pipeline actively executing via {target_script}...")
    
    if os.path.exists(target_script):
        # Executes your actual pipeline file inline so psutil can monitor it
        exec(open(target_script).read(), {'__name__': '__main__'})
    else:
        print(f"Error: Could not find script '{target_script}' in the current directory.")
        return
    # =========================================================================
    
    end_time = time.perf_counter()
    
    if resource:
        peak_mem_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    else:
        peak_mem_bytes = process.memory_info().rss
        
    duration_seconds = end_time - start_time
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)
    baseline_mem_mb = baseline_mem_bytes / (1024 * 1024)
    peak_delta_mb = peak_mem_mb - baseline_mem_mb
    
    print("\n" + "="*40)
    print("         GRAPHLENS PROFILING METRICS")
    print("="*40)
    print(f"Total Execution Time:   {duration_seconds:.4f} seconds")
    print(f"Baseline System RAM:    {baseline_mem_mb:.2f} MB")
    print(f"Peak RAM Allocation:    {peak_mem_mb:.2f} MB")
    print(f"Peak RAM Delta (Net):   {peak_delta_mb:.2f} MB")
    print("="*40)

if __name__ == "__main__":
    run_graphlens_with_profiling()
