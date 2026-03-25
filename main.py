"""
main.py — 题目提取工具

用法示例：
    python3 main.py --type proof_statement
    python3 main.py --type proof_statement --input data/ --output output/proof_results.json
    python3 main.py --list-types
"""

import argparse
import sys

from extractor import TYPE_MAPPING, extract_by_type, preprocess_results, save_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 JSON 文件中按题目类型提取题目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n支持的 --type 值：\n" + "\n".join(
            f"  {k:20s} → {v}" for k, v in TYPE_MAPPING.items()
        ),
    )
    parser.add_argument(
        "--type",
        dest="problem_type",
        required=False,
        choices=list(TYPE_MAPPING.keys()),
        help="要提取的题目类型（英文标识符）",
    )
    parser.add_argument(
        "--input",
        dest="input_dir",
        default="data",
        help="存放 JSON 文件的目录（默认：data/）",
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        default=None,
        help="结果输出路径（默认：output/<type>.json）",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="列出所有支持的题目类型并退出",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="提取后做预处理：用 problem_finally 替换 problem 并只保留核心字段（输出格式与 jsonTolean_informal 流水线兼容）",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --list-types：打印所有支持类型
    if args.list_types:
        print("支持的题目类型：")
        for key, value in TYPE_MAPPING.items():
            print(f"  --type {key:<20} 对应中文: {value}")
        sys.exit(0)

    if not args.problem_type:
        parser.print_help()
        sys.exit(1)

    chinese_type = TYPE_MAPPING[args.problem_type]
    output_file = args.output_file or f"output/{args.problem_type}.json"

    print(f"题目类型：{chinese_type}（{args.problem_type}）")
    print(f"输入目录：{args.input_dir}")
    print(f"输出文件：{output_file}")
    print("-" * 50)

    results = extract_by_type(args.input_dir, chinese_type)

    if not results:
        print(f'\n未在 {args.input_dir} 中找到任何 题目类型="{chinese_type}" 的条目。')
        sys.exit(0)

    print(f"\n共提取到 {len(results)} 条题目。")

    if args.preprocess:
        results = preprocess_results(results)
        print(f"已完成预处理，输出 {len(results)} 条（problem 字段已替换为 problem_finally，仅保留核心字段）。")

    save_results(results, output_file)


if __name__ == "__main__":
    main()
