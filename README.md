# problem_extraction

从结构化 JSON 文件中按题目类型批量提取数学题目的命令行工具，面向数学竞赛 / 教材习题 benchmark 构建场景。

---

## 目录结构

```
problem_extraction/
├── main.py               # 命令行入口
├── extractor.py          # 核心提取逻辑
├── config.json           # API 配置（本地，不入库）
├── config.example.json   # 配置模板
├── settings.json         # 其他运行参数
├── data/                 # 输入 JSON 文件（示例：bv_cvxbook_extra_exercises）
├── output/               # 提取结果输出目录
├── example/              # 示例 JSON 文件
└── src/
    ├── jsonTolean.py     # JSON → Lean 转换工具
    ├── prompt/           # LLM 提示词模板
    └── stdjson/
        └── concise_to_lean.py  # 将 problem 字段改写为 Lean 友好格式
```

---

## JSON 数据格式

`data/` 目录中每个 JSON 文件是一个题目列表，每条记录的字段如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `index` | int | 题目序号 |
| `source_idx` | string | 原始来源编号（如 "Exercise 1.1"） |
| `source` | string | 来源名称（如 "bv_cvxbook_extra_exercises"） |
| `题目类型` | string \| list | 题目类型标签（见下表） |
| `预估难度` | string \| list | 题目难度估计 |
| `problem` | string | 原始题目陈述（LaTeX 格式） |
| `proof` | string | 证明过程（可为空） |
| `direct_answer` | string | 直接答案（可为空） |
| `problem_with_context` | string | 含上下文的题目描述 |
| `problem_standardized_math` | string | 数学标准化后的题目表述 |
| `problem_finally` | string | 最终规范化版本，适合形式化 |

---

## 支持的题目类型

| `--type` 参数 | 对应中文 |
|---------------|---------|
| `proof_statement` | 证明题 |
| `calculation` | 计算题 |
| `fill_blank` | 填空题 |
| `multiple_choice` | 选择题 |
| `short_answer` | 简答题 |
| `application` | 应用题 |

---

## 安装依赖

```bash
pip install openai
```

---

## 配置

复制 `config.example.json` 为 `config.json` 并填写 API 信息：

```json
{
  "api_key": "YOUR_API_KEY",
  "base_url": "https://your-api-base/v1",
  "model": "your-model"
}
```

---

## 用法

### 列出所有支持的题目类型

```bash
python3 main.py --list-types
```

### 按题目类型提取

```bash
# 从 data/ 目录提取所有证明题，输出到 output/proof_statement.json
python3 main.py --type proof_statement

# 指定输入目录和输出文件
python3 main.py --type calculation --input data/ --output output/calc_results.json
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--type` | 必填 | 题目类型（英文标识符） |
| `--input` | `data/` | 存放 JSON 文件的目录 |
| `--output` | `output/<type>.json` | 结果输出路径 |
| `--list-types` | — | 打印所有支持的类型后退出 |

---

## 输出格式

提取结果为 JSON 数组，每条记录即原始 JSON 中匹配的完整对象，保留所有字段，方便下游处理（如 Lean 形式化）。

---

## 数据来源

`data/` 目录中预置了来自 **Boyd & Vandenberghe《Convex Optimization》补充习题**（`bv_cvxbook_extra_exercises`）的结构化题目，涵盖第 1–17 章。
