json2lean/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config.json                # API 配置（不提交 Git）
├── config.example.json        # 配置模板
├── .gitignore
├── prompts/                   # Prompt 模板目录
│   ├── concise_to_lean.md     # 预处理 prompt
│   ├── json_to_lean.md        # JSON → Lean 翻译 prompt
│   └── recovery.md            # 修复 prompt
├── src/
│   └── json2lean/
│       ├── __init__.py
│       ├── __main__.py        # python -m json2lean 入口
│       ├── main.py            # 管线编排 + CLI
│       ├── loader.py          # 读取 JSON / config / prompt 文件
│       ├── models.py          # 数据结构定义（Exercise, PipelineConfig 等）
│       ├── parser.py          # 从 JSON 提取 Exercise 对象
│       ├── preprocessor.py    # 预处理（合并 concise_to_lean，可开关）
│       ├── comment_builder.py # 生成 Lean 注释头
│       ├── translater.py      # 调用 LLM 翻译为 Lean 代码
│       ├── writer.py          # 写出 .lean 文件
│       ├── validator.py       # 编译验证（合并 interact.py）
│       ├── recover.py         # 自动修复循环（验证→修复→重编译）
│       ├── api_client.py      # OpenAI 封装 + token 用量追踪
│       └── lean_env.py        # Lean 4 环境检测与帮助信息
├── outputs/                   # 生成的 .lean 文件
└── logs/                      # token 使用日志
