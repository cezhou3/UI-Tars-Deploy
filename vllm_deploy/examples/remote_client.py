"""
Remote client example for connecting to vLLM server deployed on Slurm cluster
This script demonstrates how to connect to a remotely deployed vLLM server
with API key authentication and handle both text and multimodal queries.
"""

import argparse
import json
import requests
import base64
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union

class RemoteVLLMClient:
    def __init__(
        self, 
        host: str, 
        port: int, 
        api_key: str,
        model: str = "UI-TARS-7B-DPO",
        use_ssl: bool = False,
        timeout: int = 120
    ):
        """
        Initialize a client for connecting to remote vLLM server.
        
        Args:
            host: Server hostname or IP address
            port: Server port
            api_key: API key for authentication
            model: Model name to use
            use_ssl: Whether to use HTTPS instead of HTTP
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.protocol = "https" if use_ssl else "http"
        self.base_url = f"{self.protocol}://{self.host}:{self.port}"
        
        # Format API key for header
        if not self.api_key.startswith("Bearer "):
            if not self.api_key.startswith("sk-"):
                self.api_key = f"Bearer sk-{self.api_key}"
            else:
                self.api_key = f"Bearer {self.api_key}"
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key
        }
        
    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image file to base64 string."""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def check_connection(self) -> bool:
        """Check if the server is reachable and auth is valid."""
        try:
            url = f"{self.base_url}/v1/models"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                print(f"✅ Connection successful to {self.base_url}")
                return True
            elif response.status_code == 401:
                print(f"❌ Authentication failed. Check your API key.")
                return False
            else:
                print(f"❌ Server returned status code {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
            
    def generate_text(
        self, 
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text from a text prompt.
        
        Args:
            prompt: Text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Response from the server
        """
        url = f"{self.base_url}/v1/completions"
        data = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        print(f"Sending text request to {url}")
        print(f"Prompt: {prompt}")
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=self.headers, json=data, timeout=self.timeout)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"Request completed in {end_time - start_time:.2f} seconds")
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return {"error": response.text}
        except Exception as e:
            print(f"Request failed: {e}")
            return {"error": str(e)}
    
    def generate_multimodal(
        self,
        prompt: str,
        image_path: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text from a multimodal prompt with image.
        
        Args:
            prompt: Text prompt
            image_path: Path to image file
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response from the server
        """
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return {"error": f"Image not found: {image_path}"}
        
        # Encode image
        image_b64 = self._encode_image(image_path)
        if not image_b64:
            return {"error": "Failed to encode image"}
        
        url = f"{self.base_url}/v1/completions"
        data = {
            "model": self.model,
            "prompt": f"<img>{image_b64}</img>\n{prompt}",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        print(f"Sending multimodal request with image: {image_path}")
        print(f"Prompt: {prompt}")
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=self.headers, json=data, timeout=self.timeout)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"Request completed in {end_time - start_time:.2f} seconds")
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return {"error": response.text}
        except Exception as e:
            print(f"Request failed: {e}")
            return {"error": str(e)}

def parse_args():
    parser = argparse.ArgumentParser(description="Remote client for vLLM server")
    parser.add_argument("--host", type=str, required=True, help="Server hostname or IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--api-key", type=str, required=True, help="API key for authentication")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Text prompt")
    parser.add_argument("--image", type=str, default=None, help="Optional image path for multimodal requests")
    parser.add_argument("--ssl", action="store_true", help="Use HTTPS instead of HTTP")
    return parser.parse_args()

def main():
    """Command-line example of remote client usage."""
    args = parse_args()
    
    # Initialize client
    client = RemoteVLLMClient(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        use_ssl=args.ssl
    )
    
    # Check if server is accessible
    if not client.check_connection():
        print("Exiting due to connection or authentication issues.")
        sys.exit(1)
    
    # Generate response
    if args.image:
        result = client.generate_multimodal(args.prompt, args.image)
    else:
        result = client.generate_text(args.prompt)
    
    # Print response
    if "error" not in result:
        print("\nServer response:")
        print(result["choices"][0]["text"])
    else:
        print(f"Failed to get response: {result['error']}")

if __name__ == "__main__":
    main()
