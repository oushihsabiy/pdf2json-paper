# Copilot Change Log

## 2026-03-16

### 变更任务
- 在 src/paper 中创建新的 mdTotex.py，用于将 pdfTomd.py 产出的 Markdown 转为可编译 LaTeX。

### 具体修改内容
- 新建 [src/paper/mdTotex.py](src/paper/mdTotex.py)，实现命令行接口：python mdTotex.py input.md output.tex。
- 按模块实现函数：read_markdown、parse_blocks、convert_headings、convert_equations、convert_lists、convert_images、convert_tables、convert_math_environments、generate_latex_document、write_tex。
- 增加完整文档模板：article + amsmath/amssymb/amsthm/graphicx/hyperref/booktabs，并定义 theorem/lemma/proposition/corollary/definition 环境。
- 支持核心 Markdown 结构转换：标题、粗体/斜体、行内公式、块公式、列表、引用、图片、代码块、表格（booktabs 风格）。
- 增加数学结构识别：Theorem、Lemma、Proposition、Definition、Corollary、Proof。
- 对 theorem 类段落中的内嵌 Proof. 做外置化处理，输出独立 \begin{proof} ... \end{proof}。
- 增加健壮性处理：输入输出后缀校验、异常捕获、无法识别结构最小改写保留。

### 影响范围
- 新增文件 [src/paper/mdTotex.py](src/paper/mdTotex.py)。
- 更新记录文件 [COPILOT_CHANGES.md](COPILOT_CHANGES.md)。

## 2026-03-17

### 变更任务
- 调整简化流程第 2 步（stmt 识别）：仅当候选声明行被 Markdown 粗体包裹（`**...**`）时，才判定为 `stmt`。

### 具体修改内容
- 修改 [src/paper/mdTotex.py](src/paper/mdTotex.py) 的 `_normalize_stmt_line`：
	- 先匹配整行为 `**...**`（`^\*\*(.+?)\*\*$`）。
	- 若不满足粗体包裹，直接返回非 stmt。
	- 仅对粗体内部文本再执行 `_STMT_START_RE` 的 Theorem/Lemma/Proposition/Corollary/Definition/Remark/Assumption/Conjecture 识别。

### 影响范围
- 更新文件 [src/paper/mdTotex.py](src/paper/mdTotex.py)。
- 更新记录文件 [COPILOT_CHANGES.md](COPILOT_CHANGES.md)。

### 补充：display-math 抽取规则收紧
- 在 `replace_display_math_with_placeholders` 增加 `$$...$$` 抽取前置校验：
	- 仅当 `$$` 与内容“紧贴”时才抽取为 math block。
	- 若 `$$` 后立刻空行或 `$$` 前存在空行（即边界不紧贴），保持原文，不作为 display-math 占位符抽取。
## 2026-03-17（续）

### 变更任务
- 改进 display-math 抽取机制：从严格的"紧邻边界"规则改为简单的"配对规则"。

### 具体修改内容
- 完全重写 [src/paper/mdTotex.py](src/paper/mdTotex.py) 的 `replace_display_math_with_placeholders` 函数：
	- **新规则**：`$$` 按简单配对方式抽取 —— 第1个 `$$` 与第2个 `$$` 组成公式块，第3个与第4个组成公式块，依此类推。
	- **优势**：避免了复杂的边界检查（如"是否存在空白行"），从而支持多行矩阵、分段公式等包含内部空白行的合法数学结构。
	- **兼容性**：同时保留对 `\[...\]` 和 `\begin{equation|align|...}\end{...}` 块的正则表达式匹配。
	- **重复消除**：对重叠的块进行去重，保留最先匹配的块。
- 移除不再使用的 `_is_tight_dollar_block` 函数。

### 影响范围
- 更新文件 [src/paper/mdTotex.py](src/paper/mdTotex.py)。
- 更新记录文件 [COPILOT_CHANGES.md](COPILOT_CHANGES.md)。

### 测试验证
- 通过单元测试验证：
	- 简单的配对 `$$` 块可正确抽取。
	- 包含多行矩阵、`\begin{bmatrix}`、`\tag{2}` 等结构的复杂 `$$` 块现在作为整体保留。
	- 混合 `\[...\]`、`\begin{...}\end{...}` 和 `$$...$$` 块的文档正确处理。

## 2026-03-17（续续）

### 变更任务
- 添加 LaTeX 包以支持 `\begin{equation}...\end{equation}` 环境。

### 具体修改内容
- 在 [src/paper/mdTotex.py](src/paper/mdTotex.py) 的导言区添加 `\usepackage{mathtools}`
- 在 [src/book/mdTotex.py](src/book/mdTotex.py) 的导言区添加 `\usepackage{mathtools}`

### 说明
- `amsmath` 包（已有）原生支持 `\begin{equation}...\end{equation}` 环境
- `mathtools` 包是 `amsmath` 的扩展，提供额外功能：
	- 增强的分隔符处理（`\DeclarePairedDelimiter`, `\lparen`, `\rparen` 等）
	- 更灵活的对齐和标签处理
	- 支持非对称的 `\bigg` 和其他大小修饰符
	- 改进的 equation numbering 和引用

### 影响范围
- 更新文件 [src/paper/mdTotex.py](src/paper/mdTotex.py) 
- 更新文件 [src/book/mdTotex.py](src/book/mdTotex.py)
- 更新记录文件 [COPILOT_CHANGES.md](COPILOT_CHANGES.md)

## 2026-03-17（泄漏规则调整）

### 变更任务
- 删除将 `<<<PROOF>>>` / `<<<REST>>>` 视为提示词泄漏特征的规则。

### 具体修改内容
- 修改 [src/paper/mdTotex.py](src/paper/mdTotex.py) 的 `_has_prompt_leak`：
	- 从 `tell_tale` 列表中移除 `<<<proof>>>` 和 `<<<rest>>>`。

### 影响范围
- 更新文件 [src/paper/mdTotex.py](src/paper/mdTotex.py)
- 更新记录文件 [COPILOT_CHANGES.md](COPILOT_CHANGES.md)

## 2026-03-17（proof_split 简化）

### 变更任务
- 简化 proof 处理流程：不再拆分 `proof/rest`，proof 块整体转换。

### 具体修改内容
- 修改 [src/paper/mdTotex.py](src/paper/mdTotex.py) 的 `convert_blocks_to_latex`：
	- 删除 `blk.kind == "proof"` 分支中对 `markdown_proof_split_to_latex` 的调用。
	- 删除 proof-split 失败回退分支。
	- 统一改为对 proof 块直接执行一次 `markdown_to_latex`，并再做 `insert_block_sentinels`。

### 行为变化
- 之前：尝试将 proof 块切分为 `<<<PROOF>>>` 与 `<<<REST>>>` 两段。
- 现在：不做二次切分，整块按 proof 内容处理。

### 影响范围
- 更新文件 [src/paper/mdTotex.py](src/paper/mdTotex.py)
- 更新记录文件 [COPILOT_CHANGES.md](COPILOT_CHANGES.md)

## 2026-03-17（清理无用 proof_split 代码）

### 变更任务
- 删除已不再使用的 proof_split 函数和 prompt。

### 具体修改内容
- 修改 [src/paper/mdTotex.py](src/paper/mdTotex.py)：
	- 删除未使用的 `PROOF_SPLIT_PROMPT`。
	- 删除未使用的 `_parse_proof_split_response`。
	- 删除未使用的 `_split_proof_output`。
	- 删除未使用的 `markdown_proof_split_to_latex`。

### 影响范围
- 更新文件 [src/paper/mdTotex.py](src/paper/mdTotex.py)
- 更新记录文件 [COPILOT_CHANGES.md](COPILOT_CHANGES.md)