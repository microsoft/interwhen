#!/usr/bin/env python3
"""
Job scheduler for running bestofk_baseline.py experiments sequentially.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Base command
BASE_CMD = "python /data/b-pchanda/interwhen/examples/TTSwithVerification/bestofk_baseline.py"

# Define your experiment configurations
EXPERIMENTS = [
    # Maze experiments
    {
        "name": "maze_k4_critic",
        "args": "--task maze --k 4 --use_critic"
    },
    {
        "name": "maze_k4_critic_earlystop",
        "args": "--task maze --k 4 --use_critic --critic_early_stop"
    },
    {
        "name": "maze_k4_no_critic",
        "args": "--task maze --k 4"
    },
    
    # Game24 experiments
    {
        "name": "game24_k4_critic",
        "args": "--task game24 --k 4 --use_critic"
    },
    {
        "name": "game24_k4_critic_earlystop",
        "args": "--task game24 --k 4 --use_critic --critic_early_stop"
    },
    {
        "name": "game24_k4_no_critic",
        "args": "--task game24 --k 4"
    },
    
    # Spatialmap experiments
    {
        "name": "spatialmap_k4_critic",
        "args": "--task spatialmap --k 4 --use_critic"
    },
    {
        "name": "spatialmap_k4_critic_earlystop",
        "args": "--task spatialmap --k 4 --use_critic --critic_early_stop"
    },
    {
        "name": "spatialmap_k4_no_critic",
        "args": "--task spatialmap --k 4"
    },
]


def run_experiment(exp_config, exp_num, total_exps):
    """Run a single experiment."""
    name = exp_config["name"]
    args = exp_config["args"]
    
    print("\n" + "="*80)
    print(f"Experiment [{exp_num}/{total_exps}]: {name}")
    print(f"Command: {BASE_CMD} {args}")
    print("="*80)
    
    start_time = time.time()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Run the command
        result = subprocess.run(
            f"{BASE_CMD} {args}",
            shell=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed = time.time() - start_time
        end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if result.returncode == 0:
            print(f"\n✓ Experiment '{name}' completed successfully")
            print(f"  Started:  {start_ts}")
            print(f"  Finished: {end_ts}")
            print(f"  Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            return True, elapsed
        else:
            print(f"\n✗ Experiment '{name}' failed with exit code {result.returncode}")
            print(f"  Duration: {elapsed:.1f}s")
            return False, elapsed
            
    except KeyboardInterrupt:
        print(f"\n\n⚠ Experiment '{name}' interrupted by user")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Experiment '{name}' failed with exception: {e}")
        print(f"  Duration: {elapsed:.1f}s")
        return False, elapsed


def main():
    """Run all experiments sequentially."""
    print("="*80)
    print("JOB SCHEDULER - Running experiments sequentially")
    print("="*80)
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    total_start = time.time()
    
    try:
        for i, exp in enumerate(EXPERIMENTS, 1):
            success, duration = run_experiment(exp, i, len(EXPERIMENTS))
            results.append({
                "name": exp["name"],
                "success": success,
                "duration": duration
            })
            
            # Brief pause between experiments
            if i < len(EXPERIMENTS):
                print("\nWaiting 5 seconds before next experiment...")
                time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Job scheduler interrupted by user")
        
    finally:
        # Print summary
        total_elapsed = time.time() - total_start
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        print(f"\nCompleted: {len(results)}/{len(EXPERIMENTS)} experiments")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min, {total_elapsed/3600:.2f} hrs)")
        
        print("\nDetailed results:")
        for i, r in enumerate(results, 1):
            status = "✓" if r["success"] else "✗"
            print(f"  {i}. {status} {r['name']:40s} - {r['duration']:.1f}s ({r['duration']/60:.1f} min)")
        
        if failed > 0:
            print("\nFailed experiments:")
            for i, r in enumerate(results, 1):
                if not r["success"]:
                    print(f"  {i}. {r['name']}")
        
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
