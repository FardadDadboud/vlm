# Quick setup and usage
from vlm_evaluation_framework import VLMEvaluationFramework, create_example_config

framework = VLMEvaluationFramework("vlm_evaluation_config.json")

# 5. Comprehensive evaluation (research-grade)
comprehensive_results = framework.run_comprehensive_evaluation()