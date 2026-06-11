"""
Quick test script for performance monitoring module.

Run with: python -m tests.test_watch_performance
or: pytest tests/test_watch_performance.py
"""

import time

from mlwq.utils.watch_performance import PerformanceMonitor


def main():
    print("Testing Performance Monitor...")
    print("=" * 50)

    # Create monitor with fast sampling for testing
    monitor = PerformanceMonitor(
        sampling_interval=0.1,
        device="cuda:0",
        enable_plots=True,
        enable_csv=True,
    )

    print("\nStarting monitor...")
    monitor.start()

    print("Simulating work for 5 seconds...")
    for i in range(50):
        # Simulate some work
        sum(range(100000))
        time.sleep(0.1)

        if i % 10 == 0:
            print(f"  Progress: {i*10}%")

    print("\nStopping monitor...")
    monitor.stop()

    print("\nSummary:")
    summary = monitor.get_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    print(f"\nResults saved to: {monitor.run_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
