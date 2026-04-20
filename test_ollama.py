"""
Test script for Ollama integration
"""

from ollama_integration import check_ollama_availability, get_ollama_models, run_ollama_inference
from model_comparison import compare_models, get_best_model

# Test Ollama availability
print("Testing Ollama availability...")
is_available = check_ollama_availability()
print(f"Ollama available: {is_available}")

if is_available:
    # Test getting models
    print("\nAvailable Ollama models:")
    models = get_ollama_models()
    for model in models:
        print(f"  - {model['name']} ({model['size']} GB) [{model['id']}]")
    
    # Test inference with a small model
    print("\nTesting inference with tinyllama...")
    test_text = "This product is amazing and I love it!"
    result = run_ollama_inference("tinyllama:latest", test_text)
    
    print(f"Result: {result}")
    print(f"Label: {result.get('label')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Latency: {result.get('latency_ms')} ms")
    print(f"Energy: {result.get('energy_kwh')} kWh")
else:
    print("\n⚠️ Ollama is not running. Please start Ollama to test.")
    print("Run: ollama serve")
