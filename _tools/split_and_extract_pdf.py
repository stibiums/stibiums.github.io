#!/usr/bin/env python3
"""
PDF拆分与文本提取工具
- 将大PDF文件按页数拆分成多个小文件
- 提取每个PDF的文本内容到临时文本文件
- 避免单个文件过大导致读取失败
"""
import sys
from pathlib import Path

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    print("正在安装pypdf库...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    from pypdf import PdfReader, PdfWriter


def split_and_extract_pdf(input_path, output_dir, pages_per_file=15, extract_text=True):
    """
    将PDF文件拆分成多个小文件，并可选提取文本内容

    Args:
        input_path: 输入PDF文件路径
        output_dir: 输出目录
        pages_per_file: 每个文件包含的页数
        extract_text: 是否提取文本到txt文件

    Returns:
        tuple: (拆分的PDF文件列表, 提取的文本文件列表)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取PDF
    print(f"正在读取: {input_path}")
    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    print(f"总页数: {total_pages}")

    # 计算需要拆分的文件数
    num_files = (total_pages + pages_per_file - 1) // pages_per_file
    print(f"将拆分为 {num_files} 个文件，每个文件最多 {pages_per_file} 页")

    pdf_files = []
    text_files = []

    # 拆分PDF并提取文本
    for file_num in range(num_files):
        start_page = file_num * pages_per_file
        end_page = min((file_num + 1) * pages_per_file, total_pages)

        # 创建新的PDF
        writer = PdfWriter()
        text_content = []

        for page_num in range(start_page, end_page):
            page = reader.pages[page_num]
            writer.add_page(page)

            # 提取文本
            if extract_text:
                try:
                    page_text = page.extract_text()
                    text_content.append(f"\n{'='*60}\n")
                    text_content.append(f"第 {page_num + 1} 页\n")
                    text_content.append(f"{'='*60}\n")
                    text_content.append(page_text)
                except Exception as e:
                    text_content.append(f"\n[提取第 {page_num + 1} 页时出错: {e}]\n")

        # 保存PDF文件
        output_pdf = output_dir / f"{input_path.stem}_part{file_num + 1:02d}.pdf"
        with open(output_pdf, 'wb') as f:
            writer.write(f)
        pdf_files.append(output_pdf)
        print(f"已创建PDF: {output_pdf} (页 {start_page + 1}-{end_page})")

        # 保存文本文件
        if extract_text and text_content:
            output_txt = output_dir / f"{input_path.stem}_part{file_num + 1:02d}.txt"
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(''.join(text_content))
            text_files.append(output_txt)
            print(f"已创建文本: {output_txt}")

    print(f"\n拆分完成！")
    print(f"PDF文件保存在: {output_dir}")
    if text_files:
        print(f"文本文件保存在: {output_dir}")
        print(f"\n提取的文本文件列表:")
        for txt_file in text_files:
            print(f"  - {txt_file}")

    return pdf_files, text_files


def create_combined_text(text_files, output_path):
    """
    将多个文本文件合并成一个完整的文本文件

    Args:
        text_files: 文本文件列表
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for txt_file in text_files:
            with open(txt_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write('\n\n')
    print(f"\n已创建合并文本文件: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python split_and_extract_pdf.py <input_pdf> [output_dir] [pages_per_file]")
        print("示例: python split_and_extract_pdf.py input.pdf /tmp/output 15")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pdf_split"
    pages = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    pdf_files, text_files = split_and_extract_pdf(input_pdf, output_directory, pages_per_file=pages)

    # 创建合并的文本文件
    if text_files:
        combined_file = Path(output_directory) / f"{Path(input_pdf).stem}_complete.txt"
        create_combined_text(text_files, combined_file)
