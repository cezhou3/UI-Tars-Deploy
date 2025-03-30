"""
Example script to query the deployed UI-TARS-7B-DPO model
"""

import argparse
import json
import requests
import base64
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Query the deployed UI-TARS model")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--prompt", type=str, default="Hello, I'm using UI-TARS-7B.", help="Prompt to send")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Prepare the request
    url = f"http://{args.host}:{args.port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "UI-TARS-7B-DPO",
        "prompt": args.prompt,
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": False
    }
    
    print(f"Sending request to {url}")
    print(f"Prompt: {args.prompt}")
    
    # Send the request
    response = requests.post(url, headers=headers, json=data)
    
    # Process the response
    if response.status_code == 200:
        result = response.json()
        print("\nResponse:")
        print(result["choices"][0]["text"])
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
