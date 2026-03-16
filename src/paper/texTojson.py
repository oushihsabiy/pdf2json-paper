import re
import json
import os
import requests
from openai import OpenAI

def extract_math_environments(latex_content):
    """
    Extract definition, lemma, proposition, theorem, corollary, proof, solution from LaTeX content.
    Associate proofs with the preceding theorem/proposition.
    Returns list with start/end positions.
    """
    environments = ['definition', 'lemma', 'proposition', 'theorem', 'corollary']
    extracted = []

    # Use finditer to get positions for environments
    pattern = r'\\begin\{(' + '|'.join(environments) + r')\}(.*?)\\end\{\1\}'
    for match in re.finditer(pattern, latex_content, re.DOTALL):
        env_type = match.group(1)
        env_content = match.group(2).strip()
        start = match.start()
        end = match.end()
        extracted.append({
            "start": start,
            "end": end,
            "type": env_type,
            "content": env_content,
            "proof": ""
        })

    #TODO: 1. 对于\solution也添加到"proof"中
    #TODO: 2. 修改handle proofs的代码，要求其对proof和solution进行匹配，匹配合适的theorem/corollary/propositon/lemma
    # Handle proofs
    proof_pattern = r'\\begin\{proof\}(.*?)\\end\{proof\}'
    for match in re.finditer(proof_pattern, latex_content, re.DOTALL):
        proof_content = match.group(1).strip()
        proof_start = match.start()
        # Find the preceding theorem by checking positions
        for item in reversed(extracted):
            if item["type"] in environments and item["proof"] == "" and item["end"] < proof_start:
                item["proof"] = proof_content
                break

    return extracted

def call_llm(prompt, api_key, base_url, model):
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        return content
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return None

#TODO: 3. 修改extract_implicit_definitions函数，要求不能在theorem/lemma/proposition/corollary
#       板块中提取definition
def extract_implicit_definitions(latex_content, api_key, base_url, model):
    prompt = f"""
请从以下 LaTeX 文本中提取所有隐式的定义（definition），包括没有明确标记为 definition 的部分，以及在其他环境中（如 proposition、theorem）中包含定义性内容的段落。
隐式的定义可能以 "Let us define"、"We define"、"Define"、"Let us identify"、"We write ... as" 等开头的句子或段落，或者描述概念、符号、函数、比率等的定义性内容。

例如：
- "Let us identify p with its coefficient vector" 这样的句子。
- "We write the Crouzeix ratio as f(c,A) = τ(c,A)/β(c,A)" 这样的定义。
- "We can rewrite τ as τ(c,A) = max{{|q(c,z)|:z ∈ W(A)}}" 这样的重写定义。

返回 JSON 格式的列表，每个元素包含：
- "content": 定义内容（包括相关数学表达式）
- "proof": 如果有相关证明则填写，否则空字符串
- "type": "definition"
- "position": 定义在文本中开始的字符位置（从0开始）
- "预估难度": ""
- "source": ""
- "source_index": ""

请确保提取全面，不要遗漏任何潜在的定义，按照它们在文本中出现的顺序排列。
如果没有找到隐式定义，返回空列表 []。

文本：
{latex_content}

请只返回 JSON 列表，不要其他内容。
"""
    try:
        response = call_llm(prompt, api_key, base_url, model)
        if response is None:
            return []
        # 尝试解析 JSON
        implicit_defs = json.loads(response)
        # 转换为统一格式
        unified = []
        for item in implicit_defs:
            position = item.get("position", 0)
            unified.append({
                "start": position,
                "end": position + len(item.get("content", "")),
                "type": item.get("type", "definition"),
                "content": item.get("content", ""),
                "proof": item.get("proof", ""),
                "预估难度": item.get("预估难度", ""),
                "source": item.get("source", ""),
                "source_index": item.get("source_index", "")
            })
        return unified
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        print(f"LLM 响应: {response}")
        return []
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return []

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract theorem-like environments from a LaTeX file into JSON.")
    parser.add_argument("input_tex", help="Path to the input .tex file")
    parser.add_argument("output_json", help="Path for the extracted JSON output")
    args = parser.parse_args()

    latex_file = args.input_tex
    output_file = args.output_json

    if not os.path.exists(latex_file):
        print(f"File {latex_file} does not exist.")
        return

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    api_key = config['api_key']
    base_url = config['base_url']
    model = config['model']

    with open(latex_file, 'r', encoding='utf-8') as f:
        latex_content = f.read()

    print("开始提取显式数学环境...")
    extracted_data = extract_math_environments(latex_content)
    print(f"提取到 {len(extracted_data)} 个显式环境。")

    print("开始提取隐式定义（使用 LLM）...")
    implicit_defs = extract_implicit_definitions(latex_content, api_key, base_url, model)
    print(f"提取到 {len(implicit_defs)} 个隐式定义。")

    print("合并和排序所有项目...")
    all_items = extracted_data + implicit_defs
    all_items.sort(key=lambda x: x["start"])
    print(f"总共 {len(all_items)} 个项目。")

    print("分配索引并格式化...")
    for i, item in enumerate(all_items, start=1):
        item["index"] = i
        item["problem"] = item["content"]
        item["题目类型"] = item["type"]
        # Remove unnecessary keys
        del item["start"], item["end"], item["content"], item["type"]

    extracted_data = all_items

    print("保存到输出文件...")
    # Ensure output directory exists
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)

    print(f"提取完成！数据已保存到 {output_file}")

if __name__ == "__main__":
    main()