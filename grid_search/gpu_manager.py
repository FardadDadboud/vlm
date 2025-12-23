"""
GPU Manager for dynamic GPU allocation across experiments
"""

import subprocess
import re
from typing import List, Optional, Dict


class GPUManager:
    """Manages GPU allocation for parallel experiments"""
    
    def __init__(self, safety_margin_gb: float = 1.0):
        """
        Initialize GPU manager
        
        Args:
            safety_margin_gb: Reserved memory per GPU (GB)
        """
        self.safety_margin_gb = safety_margin_gb
        self.available_gpus = self._detect_gpus()
        self.gpu_capacities = self._get_gpu_memory()
        self.gpu_allocated = {gpu_id: 0.0 for gpu_id in self.available_gpus}
        
        print(f"GPU Manager initialized:")
        print(f"  Available GPUs: {self.available_gpus}")
        print(f"  GPU Capacities: {self.gpu_capacities}")
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse output to get GPU IDs
            gpu_ids = []
            for line in result.stdout.strip().split('\n'):
                match = re.match(r'GPU (\d+):', line)
                if match:
                    gpu_ids.append(int(match.group(1)))
            return gpu_ids
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: nvidia-smi not available, using CPU")
            return []
    
    def _get_gpu_memory(self) -> Dict[int, float]:
        """Get total memory for each GPU in GB"""
        gpu_memory = {}
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.strip().split('\n'):
                if line:
                    gpu_id, mem_mb = line.split(',')
                    gpu_memory[int(gpu_id)] = float(mem_mb) / 1024  # Convert to GB
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return gpu_memory
    
    def allocate_gpu(self, memory_needed_gb: float) -> Optional[int]:
        """
        Find GPU with enough free memory
        
        Args:
            memory_needed_gb: Memory required in GB
            
        Returns:
            GPU ID if available, None otherwise
        """
        for gpu_id in self.available_gpus:
            capacity = self.gpu_capacities[gpu_id]
            allocated = self.gpu_allocated[gpu_id]
            free = capacity - allocated - self.safety_margin_gb
            
            if free >= memory_needed_gb:
                self.gpu_allocated[gpu_id] += memory_needed_gb
                return gpu_id
        
        return None  # All GPUs full
    
    def release_gpu(self, gpu_id: int, memory_amount_gb: float):
        """Release memory allocation on GPU"""
        if gpu_id in self.gpu_allocated:
            self.gpu_allocated[gpu_id] -= memory_amount_gb
            self.gpu_allocated[gpu_id] = max(0, self.gpu_allocated[gpu_id])
    
    def get_max_parallel_jobs(self, memory_per_job_gb: float) -> int:
        """Calculate maximum number of parallel jobs"""
        total_capacity = 0
        for gpu_id in self.available_gpus:
            capacity = self.gpu_capacities[gpu_id] - self.safety_margin_gb
            total_capacity += capacity
        
        return int(total_capacity / memory_per_job_gb)
    
    def get_status(self) -> str:
        """Get current allocation status"""
        status = "GPU Status:\n"
        for gpu_id in self.available_gpus:
            capacity = self.gpu_capacities[gpu_id]
            allocated = self.gpu_allocated[gpu_id]
            free = capacity - allocated - self.safety_margin_gb
            status += f"  GPU {gpu_id}: {allocated:.1f}/{capacity:.1f} GB allocated ({free:.1f} GB free)\n"
        return status
