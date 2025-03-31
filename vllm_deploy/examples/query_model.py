"""
Example script to query the deployed UI-TARS-7B-DPO model
"""

import argparse
import json
import requests
import base64
from pathlib import Path
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Query the deployed UI-TARS model")
    parser.add_argument("--host", type=str, default="localhost", help="Server host (IP or hostname)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--prompt", type=str, default="Hello, I'm using UI-TARS-7B.", help="Prompt to send")
    parser.add_argument("--api-key", type=str, required=True, help="API key for authentication")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (optional)")
    parser.add_argument("--ssl", action="store_true", help="Use HTTPS instead of HTTP")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    return parser.parse_args()

def encode_image(image_path):
    """Encode image to base64"""
    if not image_path:
        return None
    
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def main():
    args = parse_args()
    
    # Build URL with appropriate protocol
    protocol = "https" if args.ssl else "http"
    url = f"{protocol}://{args.host}:{args.port}/v1/completions"
    
    # Set up headers with API key authentication
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        if args.api_key.startswith("sk-"):
            headers["Authorization"] = f"Bearer {args.api_key}"
        else:
            headers["Authorization"] = f"Bearer sk-{args.api_key}"
    
    # Base prompt
    prompt_data = args.prompt
    
    # If image is provided, format as multimodal prompt
    if args.image and Path(args.image).exists():
        image_b64 = encode_image(args.image)
        if image_b64:
            data = {
                "model": "UI-TARS-7B-DPO",
                "prompt": f"<img>{image_b64}</img>\n{prompt_data}",
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": False
            }
            print(f"Sending multimodal request with image: {args.image}")
        else:
            print("Failed to encode image, sending text-only request")
            data = {
                "model": "UI-TARS-7B-DPO",
                "prompt": prompt_data,
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": False
            }
    else:
        data = {
            "model": "UI-TARS-7B-DPO",
            "prompt": prompt_data,
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": False
        }
    
    print(f"Connecting to server: {url}")
    print(f"Prompt: {args.prompt}")
    print("API Authentication: Enabled")
    
    try:
        # Send the request with timeout
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=args.timeout)
        end_time = time.time()
        
        # Process the response
        if response.status_code == 200:
            result = response.json()
            print("\nResponse:")
            print(result["choices"][0]["text"])
            print(f"\nRequest completed in {end_time - start_time:.2f} seconds")
        elif response.status_code == 401:
            print("Error: Authentication failed. Please check your API key.")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {url}")
        print("Please check the hostname/IP and port, and ensure the server is running.")
    except requests.exceptions.Timeout:
        print(f"Error: Request timed out after {args.timeout} seconds")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
