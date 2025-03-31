# 从本地连接到远程vLLM服务详细指南

本指南详细介绍如何从本地机器安全地连接到部署在Slurm集群或其他云服务器上的vLLM模型服务。

## 目录

1. [前提条件](#前提条件)
2. [获取连接信息](#获取连接信息)
3. [安装必要依赖](#安装必要依赖)
4. [连接方法](#连接方法)
   - [使用remote_client.py](#使用remote_client脚本)
   - [使用query_model.py](#使用query_model脚本)
   - [使用curl命令](#使用curl命令)
   - [在Python代码中连接](#在python代码中连接)
   - [使用应用程序连接](#使用应用程序连接)
5. [特殊使用场景](#特殊使用场景)
   - [多模态请求](#多模态请求)
   - [流式输出](#流式输出)
   - [批量处理](#批量处理)
6. [故障排除](#故障排除)
7. [安全最佳实践](#安全最佳实践)
8. [与其他框架集成](#与其他框架集成)

## 前提条件

在开始连接之前，确保您具备以下条件：

- 已成功部署的vLLM服务（通过Slurm或其他方式）
- 服务器的IP地址或主机名
- 服务端口号（默认为8000）
- 有效的API密钥
- 本地安装的Python 3.7+环境（如需使用Python客户端）

## 获取连接信息

### Slurm部署的服务器信息

当使用`run_slurm.sh`部署vLLM服务时，系统会自动生成一个`server_connection_info.txt`文件。该文件包含以下关键信息：

```
Server URL: http://10.123.45.67:8000
API Key: sk-abcdefghijklmnopqrstuvwxyz123456
Example curl command:
curl -X POST "http://10.123.45.67:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz123456" \
  -d '{"model":"UI-TARS-7B-DPO", "prompt":"Hello, world", "max_tokens":100}'
```

在连接到服务器之前，您需要：

1. 记下服务器URL（包括IP和端口）
2. 保存API密钥，它用于身份验证
3. 确认您的本地机器能够通过网络访问该服务器IP和端口

> **注意**：如果服务器位于防火墙后面或在不同的网络中，可能需要设置VPN、SSH隧道或端口转发。

### 手动查找服务器信息

如果您无法访问`server_connection_info.txt`，可以：

1. 查询运行vLLM服务的节点IP：`squeue -u $USER`
2. 使用SSH连接到该节点并运行`hostname -I`获取IP地址
3. 检查vLLM服务日志以确认使用的端口

## 安装必要依赖

在本地机器上，安装以下Python包以便与远程vLLM服务交互：

```bash
pip install requests pillow argparse
```

如果需要处理多模态查询（包含图像），还需安装：

```bash
pip install Pillow
```

## 连接方法

### 使用remote_client脚本

我们提供的`remote_client.py`是最全功能的连接工具，支持文本、多模态、流式输出以及结果保存等功能。

**基本文本查询：**

```bash
python examples/remote_client.py \
  --host 10.123.45.67 \
  --port 8000 \
  --api-key sk-abcdefghijklmnopqrstuvwxyz123456 \
  --prompt "解释量子计算的基本原理"
```

**检查服务器状态：**

```bash
python examples/remote_client.py \
  --host 10.123.45.67 \
  --port 8000 \
  --api-key sk-abcdefghijklmnopqrstuvwxyz123456 \
  --info
```

**更多参数说明：**

- `--host`: 服务器IP地址或主机名
- `--port`: 服务器端口
- `--api-key`: API密钥
- `--prompt`: 文本提示
- `--image`: 图像文件路径（多模态查询）
- `--ssl`: 使用HTTPS（如果服务器配置了SSL）
- `--stream`: 启用流式输出
- `--save-dir`: 结果保存目录
- `--max-tokens`: 生成的最大标记数
- `--temperature`: 采样温度
- `--model`: 模型名称
- `--info`: 仅获取服务器信息

### 使用query_model脚本

`query_model.py`是一个更简单的脚本，适合基本查询：

```bash
python examples/query_model.py \
  --host 10.123.45.67 \
  --port 8000 \
  --api-key sk-abcdefghijklmnopqrstuvwxyz123456 \
  --prompt "你好，请介绍一下自己"
```

### 使用curl命令

如果您更习惯使用命令行，可以直接使用curl：

```bash
curl -X POST "http://10.123.45.67:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz123456" \
  -d '{
    "model": "UI-TARS-7B-DPO",
    "prompt": "什么是深度学习？",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 在Python代码中连接

以下是将vLLM服务集成到您自己Python代码中的示例：

```python
import requests
import json

# 连接配置
server_url = "http://10.123.45.67:8000"
api_key = "sk-abcdefghijklmnopqrstuvwxyz123456"

# 准备请求
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
data = {
    "model": "UI-TARS-7B-DPO",
    "prompt": "解释强化学习的原理",
    "max_tokens": 150,
    "temperature": 0.7
}

# 发送请求
response = requests.post(
    f"{server_url}/v1/completions", 
    headers=headers, 
    json=data, 
    timeout=30
)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print("模型回复:")
    print(result["choices"][0]["text"])
else:
    print(f"错误 {response.status_code}: {response.text}")
```

### 使用应用程序连接

您也可以将vLLM服务集成到各种应用程序中：

**在Flask Web应用中：**

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
VLLM_URL = "http://10.123.45.67:8000/v1/completions"
API_KEY = "sk-abcdefghijklmnopqrstuvwxyz123456"

@app.route('/ask', methods=['POST'])
def ask_model():
    user_input = request.json.get('question', '')
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "UI-TARS-7B-DPO",
        "prompt": user_input,
        "max_tokens": 100
    }
    
    response = requests.post(VLLM_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return jsonify({"answer": result["choices"][0]["text"]})
    else:
        return jsonify({"error": f"模型请求失败: {response.status_code}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## 特殊使用场景

### 多模态请求

发送包含图像的请求（使用remote_client.py）：

```bash
python examples/remote_client.py \
  --host 10.123.45.67 \
  --port 8000 \
  --api-key sk-abcdefghijklmnopqrstuvwxyz123456 \
  --prompt "详细描述这张图片中的内容" \
  --image /path/to/your/image.jpg
```

在自定义Python代码中发送多模态请求：

```python
import requests
import base64

# 读取并编码图像
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

image_b64 = encode_image("/path/to/your/image.jpg")

# 准备请求
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-abcdefghijklmnopqrstuvwxyz123456"
}

# 注意多模态格式，UI-TARS使用<img>标签包装base64图像
data = {
    "model": "UI-TARS-7B-DPO",
    "prompt": f"<img>{image_b64}</img>\n描述这张图片",
    "max_tokens": 300,
    "temperature": 0.7
}

# 发送请求
response = requests.post(
    "http://10.123.45.67:8000/v1/completions", 
    headers=headers, 
    json=data
)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print(result["choices"][0]["text"])
```

### 流式输出

流式输出允许您在模型生成过程中实时接收结果：

```bash
python examples/remote_client.py \
  --host 10.123.45.67 \
  --port 8000 \
  --api-key sk-abcdefghijklmnopqrstuvwxyz123456 \
  --prompt "写一篇关于人工智能与人类未来的短文" \
  --stream
```

在Python代码中处理流式输出：

```python
import requests
import json

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-abcdefghijklmnopqrstuvwxyz123456"
}

data = {
    "model": "UI-TARS-7B-DPO",
    "prompt": "写一篇关于人工智能与人类未来的短文",
    "max_tokens": 500,
    "temperature": 0.7,
    "stream": True
}

response = requests.post(
    "http://10.123.45.67:8000/v1/completions", 
    headers=headers, 
    json=data, 
    stream=True
)

# 处理流式响应
if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data:'):
                line_json = line_str[5:].strip()
                if line_json == "[DONE]":
                    break
                try:
                    chunk = json.loads(line_json)
                    if len(chunk['choices']) > 0:
                        text_chunk = chunk['choices'][0].get('text', '')
                        print(text_chunk, end='', flush=True)
                except:
                    pass
    print()  # 最后打印换行
else:
    print(f"错误 {response.status_code}: {response.text}")
```

### 批量处理

处理大量请求的示例代码：

```python
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor

def query_model(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-abcdefghijklmnopqrstuvwxyz123456"
    }
    
    data = {
        "model": "UI-TARS-7B-DPO",
        "prompt": prompt,
        "max_tokens": 100
    }
    
    response = requests.post(
        "http://10.123.45.67:8000/v1/completions", 
        headers=headers, 
        json=data
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    else:
        return f"错误: {response.status_code}"

# 批量处理多个请求
prompts = [
    "什么是机器学习?",
    "解释神经网络的工作原理",
    "什么是自然语言处理?",
    # 更多提示...
]

# 使用线程池并行处理请求
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(query_model, prompts))

# 打印结果
for i, result in enumerate(results):
    print(f"问题 {i+1}: {prompts[i]}")
    print(f"回答: {result}\n")
```

## 故障排除

### 常见问题及解决方案

1. **连接被拒绝**
   - 确认服务器IP和端口是否正确
   - 检查网络连接（是否在同一网络或VPN）
   - 检查服务器防火墙设置

2. **认证失败 (401 错误)**
   - 确认API密钥格式正确（包括"sk-"前缀）
   - 检查Authorization头是否包含"Bearer "前缀
   - 尝试重新生成API密钥

3. **请求超时**
   - 增加timeout参数值
   - 检查服务器负载是否过高
   - 减少请求的max_tokens数量

4. **内存错误**
   - 尝试减小请求的图像大小（如果是多模态请求）
   - 减少max_tokens数量
   - 检查服务器GPU内存使用情况

### 检查服务器状态

使用以下命令检查服务器状态：

```bash
python examples/remote_client.py \
  --host 10.123.45.67 \
  --port 8000 \
  --api-key sk-abcdefghijklmnopqrstuvwxyz123456 \
  --info
```

检查Slurm作业状态：

```bash
squeue -u $USER
```

## 安全最佳实践

1. **API密钥管理**
   - 不要在公共代码中硬编码API密钥
   - 使用环境变量或配置文件存储API密钥
   - 定期轮换API密钥

2. **网络安全**
   - 考虑为生产环境启用HTTPS
   - 使用SSH隧道加密通信（如果不能使用HTTPS）
   - 限制IP访问范围

3. **请求限制**
   - 实施请求频率限制以防止滥用
   - 监控异常使用模式

4. **数据隐私**
   - 避免发送敏感或个人身份信息
   - 考虑在发送前对输入数据进行脱敏处理

## 与其他框架集成

### 使用与LMMEnginevLLM相同风格连接

如果您正在使用类似 Agent-S 中的 LMMEnginevLLM 架构，可以直接使用以下代码模式连接到远程vLLM服务器：

```python
import os
from openai import OpenAI

class RemotevLLMEngine:
    def __init__(self, base_url, api_key=None, model=None, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        
        # 创建OpenAI客户端，指向远程vLLM服务器
        self.llm_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(self, messages, temperature=0.0, top_p=0.8, repetition_penalty=1.05, max_new_tokens=512, **kwargs):
        """Generate the next message based on previous messages"""
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty},
        )
        return completion.choices[0].message.content

# 使用示例
engine = RemotevLLMEngine(
    base_url="http://10.123.45.67:8000",
    api_key="sk-abcdefghijklmnopqrstuvwxyz123456",
    model="UI-TARS-7B-DPO"
)

# 准备消息
messages = [
    {"role": "system", "content": [{"type": "text", "text": "你是一个有用的助手。"}]},
    {"role": "user", "content": [{"type": "text", "text": "解释什么是人工智能？"}]}
]

# 获取回复
response = engine.generate(messages)
print(response)
```

### 处理多模态消息

对于包含图像的多模态消息，格式如下：

```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 编码图像
base64_image = encode_image("path/to/image.jpg")

# 准备包含图像的消息
messages = [
    {"role": "system", "content": [{"type": "text", "text": "你是一个有用的视觉助手。"}]},
    {"role": "user", "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image", "image": f"data:image;base64,{base64_image}"}
    ]}
]

# 获取回复
response = engine.generate(messages)
print(response)
```

### 完整集成代码示例

我们提供了一个完整的集成示例脚本，展示如何使用与Agent-S相同的代码风格连接远程vLLM服务器：

```bash
python examples/agent_connect_example.py \
  --host 10.123.45.67 \
  --port 8000 \
  --api-key sk-abcdefghijklmnopqrstuvwxyz123456 \
  --model UI-TARS-7B-DPO \
  --prompt "这是什么图片？" \
  --image path/to/image.jpg \
  --system "你是一个专业的图像分析助手，可以详细描述图片内容。"
```

有关完整代码，请参考 `examples/agent_connect_example.py`。

## 其他连接示例

### 使用Jupyter Notebook连接

```python
# 在Jupyter Notebook中使用
import requests
import IPython.display as display

def ask_model(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-abcdefghijklmnopqrstuvwxyz123456"
    }
    
    data = {
        "model": "UI-TARS-7B-DPO",
        "prompt": prompt,
        "max_tokens": 200
    }
    
    response = requests.post(
        "http://10.123.45.67:8000/v1/completions", 
        headers=headers, 
        json=data
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    else:
        return f"错误 {response.status_code}: {response.text}"

# 交互式使用
prompt = "解释BERT模型的工作原理"
display.Markdown(ask_model(prompt))
```

### 使用JS/Node.js连接

```javascript
// Node.js示例
const axios = require('axios');

async function queryModel(prompt) {
    try {
        const response = await axios.post('http://10.123.45.67:8000/v1/completions', {
            model: 'UI-TARS-7B-DPO',
            prompt: prompt,
            max_tokens: 100,
            temperature: 0.7
        }, {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer sk-abcdefghijklmnopqrstuvwxyz123456'
            }
        });
        
        return response.data.choices[0].text;
    } catch (error) {
        console.error('请求错误:', error.response ? error.response.data : error.message);
        return '发生错误';
    }
}

// 使用示例
queryModel('什么是深度学习？').then(result => {
    console.log(result);
});
```

---

如有任何问题或需要进一步的帮助，请联系系统管理员或参考vLLM官方文档。
