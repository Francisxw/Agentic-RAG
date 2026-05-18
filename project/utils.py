"""
工具函数模块

提供 PDF 转 Markdown 转换、目录清理和上下文 token 估算等通用工具函数。
"""
import os
import shutil
import subprocess
import config
from pathlib import Path
import glob
import tiktoken


def clear_directory_contents(directory: Path) -> None:
    """清空目录内容（保留目录本身）

    安全删除指定目录下的所有文件和子目录，但保留目录本身。
    适用于 Docker 卷或绑定挂载根目录。

    Args:
        directory: 要清空的目录路径
    """
    directory = Path(directory)
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pdf_to_markdown(pdf_path, output_dir):
    """使用 MinerU CLI 将单个 PDF 文件转换为 Markdown 格式。

    尝试使用 MinerU hybrid-engine 进行高质量解析（Layout 分析 + OCR + VLM），
    失败时降级到 pymupdf4llm 纯文本提取。

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出 Markdown 文件的目录路径
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    target_md = (output_dir / pdf_path.stem).with_suffix(".md")

    # 已存在则跳过
    if target_md.exists():
        return

    try:
        tmp_output = output_dir / f".mineru_tmp_{pdf_path.stem}"

        result = subprocess.run(
            ["mineru", "-p", str(pdf_path), "-o", str(tmp_output),
            "-b", config.MINERU_BACKEND],
            timeout=1000
        )

        if result.returncode == 0:
            md_files = list(tmp_output.rglob("*.md"))
            if md_files:
                shutil.copy(md_files[0], target_md)
                shutil.rmtree(tmp_output, ignore_errors=True)
                return

        shutil.rmtree(tmp_output, ignore_errors=True)

    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # 降级：MinerU 不可用时走 pymupdf4llm
    _pdf_to_markdown_fallback(pdf_path, output_dir, target_md)


def _pdf_to_markdown_fallback(pdf_path, output_dir, target_md):
    """降级方案：使用 pymupdf4llm 提取 PDF 中的文本和嵌入图片。

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录路径
        target_md: 目标 Markdown 文件路径
    """
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(doc, header=False, footer=False,
        page_separators=True, ignore_images=False, write_images=True,
        image_path=str(output_dir))
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    target_md.write_bytes(md_cleaned.encode('utf-8'))


def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    """将匹配的 PDF 文件批量转换为 Markdown 格式

    根据文件路径模式查找所有 PDF 文件，逐个转换为 Markdown 并保存到
    config.MARKDOWN_DIR 目录。已存在的 Markdown 文件会被跳过，
    除非设置 overwrite=True。

    Args:
        path_pattern: 文件路径模式（如 "docs/*.pdf"）
        overwrite: 是否覆盖已存在的 Markdown 文件，默认为 False
    """
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)


def estimate_context_tokens(messages: list) -> int:
    """估算消息列表的 token 数量

    使用 tiktoken（gpt-4 编码）计算所有消息内容的 token 总数，
    用于上下文压缩决策。

    Args:
        messages: 包含 content 属性的消息对象列表

    Returns:
        估算的 token 总数
    """
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, 'content') and msg.content)
