"""
Simplified Performance Monitoring Module for MLWQ

Monitors essential metrics: GPU usage, VRAM usage, CPU usage, RAM usage.
All metrics plotted together vs execution time.
"""

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class PerformanceMonitor:
    """
    Simplified performance monitor tracking only essential metrics.
    """

    # Cache for static GPU handles
    _gpu_handle = None
    _pynvml = None

    def __init__(
        self,
        output_dir: str = "metrics/watch_performance",
        sampling_interval: float = 0.5,
        device: str = "cuda:0",
    ):
        """
        Initialize the performance monitor.

        Args:
            output_dir: Directory to save metrics
            sampling_interval: Seconds between samples (default: 0.5s)
            device: CUDA device string (e.g., 'cuda:0')
        """
        self.sampling_interval = sampling_interval
        self.device = device
        self.output_dir = Path(output_dir)

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / timestamp

        # Thread control
        self._monitoring = False
        self._thread = None
        self.start_time = None

        # Storage: [time, gpu_pct, vram_pct, cpu_pct, ram_pct]
        self._data = []

        # Initialize monitoring
        self._init_gpu()
        self._init_psutil()

    def _init_gpu(self):
        """Initialize NVIDIA GPU monitoring."""
        if PerformanceMonitor._gpu_handle is not None:
            self.gpu_handle = PerformanceMonitor._gpu_handle
            self.pynvml = PerformanceMonitor._pynvml
            self.gpu_available = True
            return

        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml

            gpu_index = int(self.device.split(":")[1]) if self.device.startswith("cuda:") else 0
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            PerformanceMonitor._gpu_handle = self.gpu_handle
            PerformanceMonitor._pynvml = pynvml
            self.gpu_available = True
        except Exception:
            self.gpu_available = False
            self.gpu_handle = None
            self.pynvml = None

    def _init_psutil(self):
        """Initialize psutil for CPU/RAM monitoring."""
        try:
            import psutil
            self.psutil = psutil
            self.cpu_available = True
        except Exception:
            self.cpu_available = False
            self.psutil = None

    def _collect_metrics(self, elapsed: float) -> list:
        """Collect all metrics in one go."""
        gpu_pct = 0.0
        vram_pct = 0.0
        cpu_pct = 0.0
        ram_pct = 0.0

        # GPU metrics
        if self.gpu_available and self.gpu_handle:
            try:
                utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_pct = float(utilization.gpu)

                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                vram_pct = (mem_info.used / mem_info.total) * 100
            except Exception:
                pass

        # CPU/RAM metrics
        if self.cpu_available:
            try:
                cpu_pct = self.psutil.cpu_percent(interval=0.01)
                ram_pct = self.psutil.virtual_memory().percent
            except Exception:
                pass

        return [elapsed, gpu_pct, vram_pct, cpu_pct, ram_pct]

    def _monitor_loop(self):
        """Main monitoring loop."""
        self.start_time = time.time()
        time_time = time.time
        sleep = time.sleep

        while self._monitoring:
            now = time_time()
            elapsed = now - self.start_time

            row = self._collect_metrics(elapsed)
            self._data.append(row)

            sleep(max(0, self.sampling_interval - (time.time() - now)))

    def start(self):
        """Start monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"Performance monitoring started (interval: {self.sampling_interval}s)")

    def stop(self):
        """Stop monitoring and save results."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=5.0)

        self._save_results()
        print(f"Results saved to: {self.run_dir}")

    def _save_results(self):
        """Save metrics and generate plot."""
        if not self._data:
            print("No data collected")
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)

        data = np.array(self._data)
        times = data[:, 0]

        # Save CSV
        try:
            import polars as pl

            df = pl.DataFrame(
                data,
                orient="row",
                schema=["time_s", "gpu_usage_pct", "vram_usage_pct", "cpu_usage_pct", "ram_usage_pct"]
            )
            df.write_csv(self.run_dir / "metrics.csv")
        except ImportError:
            # Fallback to manual CSV
            with open(self.run_dir / "metrics.csv", "w") as f:
                f.write("time_s,gpu_usage_pct,vram_usage_pct,cpu_usage_pct,ram_usage_pct\n")
                for row in self._data:
                    f.write(f"{row[0]:.4f},{row[1]:.2f},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f}\n")

        # Save JSON
        json_data = {
            "metrics": [
                {
                    "time_s": row[0],
                    "gpu_usage_pct": row[1],
                    "vram_usage_pct": row[2],
                    "cpu_usage_pct": row[3],
                    "ram_usage_pct": row[4],
                }
                for row in self._data
            ]
        }
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(json_data, f, indent=2)

        # Generate plot
        self._plot(data)

        print(f"Metrics saved: {len(self._data)} samples")

    def _plot(self, data):
        """Generate combined plot."""
        times = data[:, 0]
        gpu_pct = data[:, 1]
        vram_pct = data[:, 2]
        cpu_pct = data[:, 3]
        ram_pct = data[:, 4]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle("System Performance Metrics", fontsize=14, fontweight="bold")

        # Top plot: GPU and VRAM
        ax1 = axes[0]
        ax1.plot(times, gpu_pct, "b-", linewidth=1.5, label="GPU Usage (%)")
        ax1.plot(times, vram_pct, "r--", linewidth=1.5, label="VRAM Usage (%)")
        ax1.set_ylabel("Usage (%)")
        ax1.set_title("GPU Metrics")
        ax1.set_ylim(0, 105)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: CPU and RAM
        ax2 = axes[1]
        ax2.plot(times, cpu_pct, "g-", linewidth=1.5, label="CPU Usage (%)")
        ax2.plot(times, ram_pct, "m--", linewidth=1.5, label="RAM Usage (%)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Usage (%)")
        ax2.set_title("CPU/RAM Metrics")
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.run_dir / "metrics.png", dpi=120, bbox_inches="tight")
        plt.close()

        print("Plot generated")

    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self._data:
            return {"duration_s": 0, "samples": 0}

        data = np.array(self._data)
        duration = data[-1, 0] - data[0, 0] if len(data) > 1 else 0

        return {
            "duration_s": float(duration),
            "samples": len(self._data),
            "avg_gpu_pct": float(np.mean(data[:, 1])),
            "max_gpu_pct": float(np.max(data[:, 1])),
            "avg_vram_pct": float(np.mean(data[:, 2])),
            "max_vram_pct": float(np.max(data[:, 2])),
            "avg_cpu_pct": float(np.mean(data[:, 3])),
            "max_cpu_pct": float(np.max(data[:, 3])),
            "avg_ram_pct": float(np.mean(data[:, 4])),
            "max_ram_pct": float(np.max(data[:, 4])),
        }


class MonitorContext:
    """Context manager for performance monitoring."""

    def __init__(self, output_dir: str = None, sampling_interval: float = 0.5, device: str = "cuda:0"):
        self.monitor = PerformanceMonitor(
            output_dir=output_dir,
            sampling_interval=sampling_interval,
            device=device,
        )

    def __enter__(self):
        self.monitor.start()
        return self.monitor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()


def monitor_context(**kwargs):
    """Create a monitoring context manager."""
    return MonitorContext(**kwargs)
