import runpod
import json
import os

def handler(event):
    """
    This is the handler function that will be called by RunPod.
    """
    try:
        # Get the input from the event
        input_data = event["input"]
        
        # Extract the model name and prompt
        model_name = input_data.get("model", "financegemma")
        prompt = input_data.get("prompt", "")
        options = input_data.get("options", {})
        
        # Import ollama here to ensure it's available
        import ollama
        
        # Generate the response
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options=options
        )
        
        # Return the response
        return {
            "output": {
                "response": response["response"]
            }
        }
        
    except Exception as e:
        # Return error if something goes wrong
        return {
            "error": str(e)
        }

# Start the handler
runpod.serverless.start({"handler": handler}) 