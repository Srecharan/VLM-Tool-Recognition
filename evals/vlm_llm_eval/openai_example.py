#!/usr/bin/env python3

# Example usage of OpenAI evaluation script

import os
from openai_eval_script import evaluate_tool_with_openai

# Example tool info (like what your VLM would output)
example_tool = {
    "tool_name": "adjustable wrench",
    "primary_function": "Gripping and turning nuts and bolts",
    "required_ppe": "Safety glasses, work gloves",
    "primary_hazards": ["Pinch points", "Slipping"],
    "common_misuses": ["Using wrong size", "Using as hammer"]
}

def test_openai_evaluation():
    # You would put your actual API key here
    api_key = "your-openai-api-key-here"
    
    print("Testing OpenAI Evaluation")
    print("=" * 30)
    
    # Test the evaluation function
    scores = evaluate_tool_with_openai(example_tool, api_key)
    
    print("Example Tool Information:")
    print(f"Tool: {example_tool['tool_name']}")
    print(f"Function: {example_tool['primary_function']}")
    print(f"PPE: {example_tool['required_ppe']}")
    
    print("\nEvaluation Scores:")
    for key, value in scores.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_openai_evaluation() 