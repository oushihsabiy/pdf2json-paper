# json2lean

将 JSON 格式的优化/数学练习题转换为 Lean 4 形式化文件。

## 项目结构

```
json2lean/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config.json              # API 配置（不提交到 Git）
├── config.example.json      # 配置模板
├── .gitignore
├── prompts/                 # Prompt 模板
│   ├── concise_to_lean.md   # 预处理 prompt
│   ├── json_to_lean.md      # 翻译 prompt
│   └── recovery.md          # 修复 prompt
├── src/json2lean/
│   ├── __init__.py
│   ├── __main__.py          # python -m json2lean
│   ├── main.py              # 管线入口 + CLI
│   ├── loader.py            # 读取 JSON / config / prompt
│   ├── models.py            # 数据结构定义
│   ├── parser.py            # 从 JSON 提取 Exercise 对象
│   ├── preprocessor.py      # 预处理（合并 concise_to_lean）
│   ├── comment_builder.py   # 生成 Lean 注释头
│   ├── translater.py        # 调用 LLM 翻译为 Lean 代码
│   ├── writer.py            # 写出 .lean 文件
│   ├── validator.py         # 编译验证（合并 interact.py）
│   ├── recover.py           # 自动修复循环
│   ├── api_client.py        # OpenAI 封装 + token 追踪
│   └── lean_env.py          # Lean 环境检测
├── outputs/                 # 生成的 .lean 文件
└── logs/                    # token 使用日志
```

## 快速开始

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
# 或者
pip install -e .
```

### 2. 安装 Lean 4

```bash
# 安装 elan（Lean 版本管理器）
curl https://elan-init.trycloudflare.com/elan/elan-init.sh -sSf | sh
# macOS 也可以用 Homebrew：
# brew install elan-init

# 安装稳定版 Lean 4 工具链
elan default leanprover/lean4:stable

# 验证
lean --version
lake --version
```

### 3. 初始化 Lean 项目目录

```bash
mkdir lean && cd lean
lake init LeanProject math
lake build        # 首次构建会下载 Mathlib，可能需要较长时间
cd ..
```

### 4. 配置 API

复制 `config.example.json` 为 `config.json`，填写你的 API 密钥：

```bash
cp config.example.json config.json
# 编辑 config.json，填入 api_key、base_url、model
```

`config.json` 字段说明：

| 字段                        | 说明                               |
|-----------------------------|----------------------------------|
| `api_key`                   | API 密钥                          |
| `base_url`                  | API 基础 URL                      |
| `model`                     | 模型名称                          |
| `timeout_seconds`           | API 超时（秒）                     |
| `preprocessing.enabled`     | 是否开启预处理                     |
| `preprocessing.max_tokens`  | 预处理最大 token                   |
| `preprocessing.max_attempts`| 预处理重试次数                     |
| `translation.max_tokens`    | 翻译最大 token                     |
| `translation.max_attempts`  | 翻译重试次数                       |
| `recovery.max_tokens`       | 修复最大 token                     |
| `recovery.max_retries`      | 修复最大重试次数                   |
| `lean.toolchain_dir`        | Lean 项目目录路径                  |
| `lean.timeout_seconds`      | 编译超时（秒）                     |

### 5. 运行

```bash
# 完整管线：预处理 → 翻译 → 写文件 → 验证 → 修复
python -m json2lean input.json

# 指定输出目录
python -m json2lean input.json -o my_outputs

# 跳过预处理
python -m json2lean input.json --no-preprocess

# 跳过验证和修复
python -m json2lean input.json --no-validate

# 只翻译不修复
python -m json2lean input.json --no-recover

# 覆盖模型
python -m json2lean input.json --model gpt-4o

# 覆盖配置文件
python -m json2lean input.json --config /path/to/config.json

# 设置修复重试上限
python -m json2lean input.json --max-recovery-retries 10
```

## Pipeline 流程

```
input.json
    │
    ▼
 ┌─────────────┐
 │   parser     │  提取 Exercise 对象
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │ preprocessor │  重写 problem 为 Definition/Hypothesis/Goal（可选）
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  translater  │  调用 LLM 生成 Lean 代码
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │   writer     │  写出 .lean 文件
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  validator   │  lake env lean 编译验证
 └──────┬──────┘
        │ 失败？
        ▼
 ┌─────────────┐
 │   recover    │  LLM 修复 → 重新写出 → 重新验证（循环）
 └──────┬──────┘
        ▼
   outputs/ + logs/
```

## Token 使用日志

每次运行后，`logs/` 目录下会生成 `token_usage_<timestamp>.json`：

```json
{
  "timestamp": "20260326_120000",
  "total": {
    "prompt_tokens": 12345,
    "completion_tokens": 6789,
    "total_tokens": 19134
  },
  "calls": [
    {
      "prompt_tokens": 1000,
      "completion_tokens": 500,
      "total_tokens": 1500,
      "call_type": "preprocess",
      "exercise_label": "ex_001"
    }
  ],
  "exercises": [
    {
      "label": "ex_001",
      "status": "valid",
      "repair_attempts": 0,
      "num_errors": 0,
      "num_warnings": 1
    }
  ]
}
```

## 输入 JSON 格式

输入文件是一个 JSON 数组（或嵌套结构），其中每个练习对象需包含 `problem` 字段，
以及至少一个标识字段（`source_idx`、`source`、`proof`、`direct_answer`、`题目类型`、`预估难度`）。

示例：

```json
[
  {
    "index": 1,
    "source_idx": "ex_001",
    "source": "Boyd & Vandenberghe",
    "题目类型": "凸优化",
    "预估难度": "中等",
    "problem": "Let S = {x in R^n | Ax <= b}. Prove S is convex.",
    "proof": "",
    "direct_answer": ""
  }
]
```

## 测试与扩展

### 测试建议

- 单元测试 `parser.py`：验证不同 JSON 结构下 Exercise 的提取。
- 单元测试 `validator.py`：用已知 Lean 输出测试解析逻辑。
- 集成测试：用小型 JSON 跑完整管线（可 mock API）。
- 用 `--no-validate` 快速迭代翻译质量。

### 扩展方向

- 并行编译：多个 .lean 文件可并行调用 `lake env lean`。
- 并行 API 调用：用 `asyncio` 或 `ThreadPoolExecutor` 加速预处理/翻译。
- 自定义 prompt：在 `prompts/` 中添加新 prompt，通过 `load_prompt(name)` 加载。
- Web UI：在 `main.py` 基础上包一层 Flask/FastAPI 接口。
- 更细粒度的错误分类：在 `recover.py` 中根据错误类型选择不同修复策略。
