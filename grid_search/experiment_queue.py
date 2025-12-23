"""
Experiment queue for grid search
"""

import itertools
import json
from pathlib import Path
from typing import List, Dict, Any
import copy


class ExperimentQueue:
    """Manages queue of experiments for grid search"""
    
    def __init__(self, grid_config: Dict[str, Any]):
        """
        Initialize experiment queue
        
        Args:
            grid_config: Grid search configuration
        """
        self.grid_config = grid_config
        self.base_config = self._load_base_config()
        self.experiments = self._generate_experiments()
        self.pending = list(range(len(self.experiments)))
        self.running = []
        self.completed = []
        
        print(f"Generated {len(self.experiments)} experiments")
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        config_path = self.grid_config['base_config_path']
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\nDEBUG: Loaded base config from {config_path}")
        print(f"  detector keys: {list(config.get('detector', {}).keys())}")
        print(f"  adaptation keys: {list(config.get('adaptation', {}).keys())}")
        if 'adaptation' in config and 'params' in config['adaptation']:
            print(f"  adaptation.params keys: {list(config['adaptation']['params'].keys())}")
        return config
    
    def _generate_experiments(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        grid_params = self.grid_config['grid_params']
        
        # Extract parameter names and values
        param_names = list(grid_params.keys())
        param_values = [grid_params[name] for name in param_names]
        
        # Generate all combinations
        experiments = []
        for combination in itertools.product(*param_values):
            # Create experiment config
            exp_config = copy.deepcopy(self.base_config)
            
            # Override with grid parameters
            param_dict = dict(zip(param_names, combination))
            exp_config = self._apply_params(exp_config, param_dict)
            
            # Override with dataset subset
            if 'dataset_subset' in self.grid_config:
                subset = self.grid_config['dataset_subset']
                exp_config['filters'] = subset.get('filters', exp_config['filters'])
                exp_config['max_samples'] = subset.get('max_samples', exp_config['max_samples'])
            
                        
            experiments.append({
                'id': len(experiments),
                'params': param_dict,
                'config': exp_config
            })
        
        return experiments
    
    def _apply_params(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter overrides to config"""
        for param_path, value in params.items():
            if value is None:
                print(f"DEBUG WARNING: Parameter {param_path} has None value!")
            keys = param_path.split('.')
            current = config
            try:
                for key in keys[:-1]:
                    if key not in current:
                        print(f"DEBUG: Creating missing key '{key}' in config path '{param_path}'")
                        current[key] = {}
                    current = current[key]
                print(f"DEBUG: Setting {param_path} = {value} (type: {type(value).__name__})")
                current[keys[-1]] = value
            except Exception as e:
                print(f"DEBUG ERROR: Failed to set {param_path} = {value}: {e}")
                print(f"  Current config structure: {list(config.keys())}")
                raise
        return config
    
    def has_pending(self) -> bool:
        """Check if there are pending experiments"""
        return len(self.pending) > 0
    
    def get_next(self) -> Dict[str, Any]:
        """Get next experiment from queue"""
        if not self.has_pending():
            return None
        exp_id = self.pending.pop(0)
        self.running.append(exp_id)
        return self.experiments[exp_id]
    
    def mark_completed(self, exp_id: int):
        """Mark experiment as completed"""
        if exp_id in self.running:
            self.running.remove(exp_id)
        self.completed.append(exp_id)
    
    def get_progress(self) -> Dict[str, int]:
        """Get progress statistics"""
        return {
            'total': len(self.experiments),
            'pending': len(self.pending),
            'running': len(self.running),
            'completed': len(self.completed)
        }
