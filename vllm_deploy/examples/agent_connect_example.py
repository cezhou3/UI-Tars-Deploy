"""
Example showing how to connect to a remote vLLM server using the same style as
LMMEnginevLLM from engine.py and mllm.py
"""

import os
import argparse
import base64
from typing import Dict, Any, List
from openai import OpenAI

class RemotevLLMEngine:
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        """Initialize a vLLM client similar to LMMEnginevLLM."""
        assert model is not None, "model must be provided"
        self.model = model
        self.api_key = api_key

        self.base_url = base_url
        if self.base_url is None:
            raise ValueError("An endpoint URL needs to be provided")

        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        # Initialize OpenAI client with the base_url and api_key
        self.llm_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(
        self,
        messages,
        temperature=0.0,
        top_p=0.8,
        repetition_penalty=1.05,
        max_new_tokens=512,
        **kwargs
    ):
        """Generate the next message based on previous messages."""
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty},
            **kwargs
        )
        return completion.choices[0].message.content

class RemotevLLMAgent:
    def __init__(self, engine_params=None, system_prompt=None):
        """Initialize a remote vLLM agent similar to LMMAgent."""
        if engine_params is None:
            raise ValueError("engine_params must be provided")
        
        self.engine = RemotevLLMEngine(**engine_params)
        self.messages = []  # Empty messages

        if system_prompt:
            self.add_system_prompt(system_prompt)
        else:
            self.add_system_prompt("You are a helpful assistant.")

    def encode_image(self, image_content):
        """Encode image to base64 string."""
        if isinstance(image_content, str):
            with open(image_content, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            return base64.b64encode(image_content).decode("utf-8")

    def reset(self):
        """Reset conversation history."""
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    def add_system_prompt(self, system_prompt):
        """Add or update the system prompt."""
        self.system_prompt = system_prompt
        if len(self.messages) > 0:
            self.messages[0] = {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        else:
            self.messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

    def add_message(
        self,
        text_content,
        image_content=None,
        role="user",
        put_text_last=False,
    ):
        """Add a new message to the list of messages."""
        # Format like in mllm.py for vLLM
        message = {
            "role": role,
            "content": [{"type": "text", "text": text_content}],
        }

        if image_content:
            # Check if image_content is a list or a single image
            if isinstance(image_content, list):
                # If image_content is a list of images, loop through each image
                for image in image_content:
                    base64_image = self.encode_image(image)
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image;base64,{base64_image}"},
                        }
                    )
            else:
                # If image_content is a single image, handle it directly
                base64_image = self.encode_image(image_content)
                message["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image;base64,{base64_image}"}}
                )

        # Rotate text to be the last message if desired
        if put_text_last:
            text_content = message["content"].pop(0)
            message["content"].append(text_content)

        self.messages.append(message)

    def get_response(
        self,
        user_message=None,
        image=None,
        messages=None,
        temperature=0.0,
        max_new_tokens=None,
        **kwargs,
    ):
        """Generate the next response based on previous messages."""
        if messages is None:
            messages = self.messages
        if user_message:
            self.add_message(user_message, image_content=image)

        return self.engine.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Example connection to remote vLLM")
    parser.add_argument("--host", type=str, required=True, help="Server hostname or IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--api-key", type=str, help="API key for authentication")
    parser.add_argument("--ssl", action="store_true", help="Use HTTPS")
    parser.add_argument("--model", type=str, default="UI-TARS-7B-DPO", help="Model name")
    parser.add_argument("--prompt", type=str, default="Tell me about AI", help="Prompt")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="System prompt")
    return parser.parse_args()

def main():
    """Main function demonstrating the agent usage."""
    args = parse_args()
    
    # Build base URL
    protocol = "https" if args.ssl else "http"
    base_url = f"{protocol}://{args.host}:{args.port}"
    
    # Set up engine parameters
    engine_params = {
        "base_url": base_url,
        "api_key": args.api_key,
        "model": args.model,
    }
    
    # Create the agent
    agent = RemotevLLMAgent(engine_params=engine_params, system_prompt=args.system)
    
    # Add user message, optionally with image
    if args.image:
        print(f"Adding message with image: {args.image}")
        agent.add_message(args.prompt, image_content=args.image)
    else:
        agent.add_message(args.prompt)
    
    # Get response
    print("Generating response...")
    response = agent.get_response()
    
    # Print the conversation
    print("\n===== Conversation =====")
    print(f"System: {args.system}")
    print(f"User: {args.prompt}")
    if args.image:
        print(f"[Image: {args.image}]")
    print(f"\nAssistant: {response}")
    print("=======================")

if __name__ == "__main__":
    main()
