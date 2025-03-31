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
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from openai import OpenAI

class RemoteVLLMClient:
    def __init__(
        self, 
        host: str, 
        port: int, 
        api_key: str = None,
        model: str = "UI-TARS-7B-DPO",
        use_ssl: bool = False,
        timeout: int = 120,
        save_dir: Optional[str] = None,
    ):
        """
        Initialize a client for connecting to remote vLLM server using OpenAI client.
        
        Args:
            host: Server hostname or IP address
            port: Server port
            api_key: API key for authentication (optional for some deployments)
            model: Model name to use
            use_ssl: Whether to use HTTPS instead of HTTP
            timeout: Request timeout in seconds
            save_dir: Directory to save results (None = don't save)
        """
        self.host = host
        self.port = port
        self.model = model
        self.timeout = timeout
        self.save_dir = save_dir
        self.protocol = "https" if use_ssl else "http"
        self.base_url = f"{self.protocol}://{self.host}:{self.port}"
        
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory for saving results: {self.save_dir}")
        
        # Initialize OpenAI client like in LMMEnginevLLM
        self.llm_client = OpenAI(base_url=self.base_url, api_key=api_key)
        
    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image file to base64 string."""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
        
    def _save_result(self, prompt: str, response: Dict[str, Any], image_path: Optional[str] = None) -> str:
        """Save the result to a file."""
        if not self.save_dir:
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        # Prepare data to save
        save_data = {
            "prompt": prompt,
            "timestamp": timestamp,
            "response": response,
            "model": self.model,
            "server": f"{self.host}:{self.port}"
        }
        
        if image_path:
            save_data["image_path"] = image_path
            
        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
            
        return filepath
    
    def check_connection(self) -> bool:
        """Check if the server is reachable and auth is valid."""
        try:
            # Use the OpenAI client to check models
            models = self.llm_client.models.list()
            print(f"✅ Connection successful to {self.base_url}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def format_messages(
        self, 
        prompt: str,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Format messages for vLLM following OpenAI chat format."""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
            
        # Create user message
        content = [{"type": "text", "text": prompt}]
        
        # Add image if provided (using vLLM format from mllm.py)
        if image_path:
            image_b64 = self._encode_image(image_path)
            if image_b64:
                content.append({
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image;base64,{image_b64}"},
                    }
                })
        
        messages.append({"role": "user", "content": content})
        return messages
            
    def generate_text(
        self, 
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
        stream: bool = False,
        save_result: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text from a text prompt using the OpenAI client pattern.
        """
        messages = self.format_messages(prompt, system_prompt=system_prompt)
        
        print(f"Sending text request to {self.base_url}")
        print(f"Prompt: {prompt}")
        
        try:
            start_time = time.time()
            
            # Use the OpenAI client to create completions
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body={"repetition_penalty": repetition_penalty},
                stream=stream
            )
            
            if stream:
                # Handle streaming response
                print("\nStreaming response:")
                full_text = ""
                for chunk in completion:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_text += content
                            print(content, end="", flush=True)
                
                print("\n")
                end_time = time.time()
                print(f"Request completed in {end_time - start_time:.2f} seconds")
                
                # Construct a result dict
                result = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model,
                    "choices": [{"message": {"content": full_text}, "index": 0, "finish_reason": "stop"}]
                }
                
                if save_result and self.save_dir:
                    filepath = self._save_result(prompt, result)
                    print(f"Result saved to {filepath}")
                    
                return result
            else:
                # Handle regular response
                end_time = time.time()
                result_text = completion.choices[0].message.content
                
                # Construct a result dict
                result = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model,
                    "choices": [{"message": {"content": result_text}, "index": 0, "finish_reason": "stop"}]
                }
                
                print(f"Request completed in {end_time - start_time:.2f} seconds")
                
                if save_result and self.save_dir:
                    filepath = self._save_result(prompt, result)
                    print(f"Result saved to {filepath}")
                    
                return result
        except Exception as e:
            print(f"Request failed: {e}")
            return {"error": str(e)}
    
    def generate_multimodal(
        self,
        prompt: str,
        image_path: str,
        system_prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
        save_result: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text from a multimodal prompt with image using the OpenAI client pattern.
        """
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return {"error": f"Image not found: {image_path}"}
        
        # Format messages using vLLM multimodal format
        messages = self.format_messages(prompt, image_path=image_path, system_prompt=system_prompt)
        
        print(f"Sending multimodal request with image: {image_path}")
        print(f"Prompt: {prompt}")
        
        try:
            start_time = time.time()
            
            # Use the OpenAI client to create completions
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body={"repetition_penalty": repetition_penalty}
            )
            
            end_time = time.time()
            result_text = completion.choices[0].message.content
            
            # Construct a result dict
            result = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model,
                "choices": [{"message": {"content": result_text}, "index": 0, "finish_reason": "stop"}]
            }
            
            print(f"Request completed in {end_time - start_time:.2f} seconds")
            
            if save_result and self.save_dir:
                filepath = self._save_result(prompt, result, image_path)
                print(f"Result saved to {filepath}")
                
            return result
        except Exception as e:
            print(f"Request failed: {e}")
            return {"error": str(e)}
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information using the OpenAI client."""
        try:
            models = self.llm_client.models.list()
            return {
                "server": f"{self.host}:{self.port}",
                "connection": "connected",
                "models": [model.model_dump() for model in models.data],
            }
        except Exception as e:
            return {
                "server": f"{self.host}:{self.port}",
                "connection": "error",
                "error": str(e)
            }

def parse_args():
    parser = argparse.ArgumentParser(description="Remote client for vLLM server")
    parser.add_argument("--host", type=str, required=True, help="Server hostname or IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Text prompt")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    parser.add_argument("--image", type=str, default=None, help="Optional image path for multimodal requests")
    parser.add_argument("--ssl", action="store_true", help="Use HTTPS instead of HTTP")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty parameter")
    parser.add_argument("--model", type=str, default="UI-TARS-7B-DPO", help="Model name to use")
    parser.add_argument("--info", action="store_true", help="Only get server info and exit")
    return parser.parse_args()

def main():
    """Command-line example of remote client usage."""
    args = parse_args()
    
    # Initialize client
    client = RemoteVLLMClient(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        model=args.model,
        use_ssl=args.ssl,
        save_dir=args.save_dir
    )
    
    # Check if server is accessible
    if not client.check_connection():
        print("Exiting due to connection or authentication issues.")
        sys.exit(1)
    
    # If just looking for server info, print and exit
    if args.info:
        server_info = client.get_server_info()
        print("\nServer Information:")
        print(json.dumps(server_info, indent=2))
        sys.exit(0)
    
    # Generate response
    if args.image:
        result = client.generate_multimodal(
            prompt=args.prompt,
            image_path=args.image,
            system_prompt=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
    else:
        result = client.generate_text(
            prompt=args.prompt,
            system_prompt=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stream=args.stream
        )
    
    # Print response (if not already printed in stream mode)
    if "error" not in result and not args.stream:
        print("\nServer response:")
        print(result["choices"][0]["message"]["content"])
    elif "error" in result:
        print(f"Failed to get response: {result['error']}")

if __name__ == "__main__":
    main()
