#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: 文档处理模块
使用MineruParser解析PDF文件，提取内容并生成结构化数据
"""

import os
import json
from raganything.mineru_parser import MineruParser

def parse_document(file_path, output_dir="./output", parse_method="auto"):
    """
    解析PDF文档
    
    Args:
        file_path (str): PDF文件路径
        output_dir (str): 输出目录
        parse_method (str): 解析方法
        
    Returns:
        tuple: 解析结果
    """
    print(f"开始解析文档: {file_path}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 使用MineruParser解析PDF
        json_content, md_content = MineruParser.parse_pdf(
            pdf_path=file_path, 
            output_dir=output_dir, 
            method=parse_method
        )
        
        print(f"文档解析完成，输出目录: {output_dir}")
        
        # 检查生成的文件
        check_output_files(output_dir)
        
        return json_content, md_content 
        
    except Exception as e:
        print(f"解析文档时出错: {str(e)}")
        return None, None

def check_output_files(output_dir):
    """
    检查输出文件并显示信息
    
    Args:
        output_dir (str): 输出目录
    """
    print("\n检查输出文件:")
    
    # 遍历输出目录
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"  - {file_path} ({file_size} bytes)")
            
            # 如果是JSON文件，显示内容结构
            if file.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        print(f"    JSON包含 {len(data)} 个条目")
                    elif isinstance(data, dict):
                        print(f"    JSON包含键: {list(data.keys())}")
                except Exception as e:
                    print(f"    无法读取JSON文件: {str(e)}")

def load_parsed_content(output_dir):
    """
    加载解析后的内容
    
    Args:
        output_dir (str): 输出目录
        
    Returns:
        dict: 包含JSON和MD内容的字典
    """
    content = {}
    
    # 查找JSON和MD文件
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file.endswith('_content_list.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content['json_data'] = json.load(f)
                        content['json_path'] = file_path
                    print(f"加载JSON内容: {len(content['json_data'])} 个条目")
                except Exception as e:
                    print(f"加载JSON文件失败: {str(e)}")
                    
            elif file.endswith('.md'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content['md_data'] = f.read()
                        content['md_path'] = file_path
                    print(f"加载MD内容: {len(content['md_data'])} 个字符")
                except Exception as e:
                    print(f"加载MD文件失败: {str(e)}")
    
    return content

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PDF文档解析器')
    parser.add_argument('--input', '-i', default="./input_files/upfluencer.pdf", 
                       help='输入PDF文件路径')
    parser.add_argument('--output', '-o', default="./output", 
                       help='输出目录')
    parser.add_argument('--method', '-m', default="auto", 
                       help='解析方法 (auto/ocr/layout)')
    parser.add_argument('--db-name', '-d', 
                       help='数据库名称 (用于区分不同领域的文档)')
    
    args = parser.parse_args()
    
    # 如果指定了数据库名称，调整输出路径
    if args.db_name:
        args.output = f"./output_{args.db_name}"
        print(f"📁 使用专用输出目录: {args.output}")
    
    print("=== RAG问答系统 - Step 1: 文档处理 ===")
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"解析方法: {args.method}")
    if args.db_name:
        print(f"数据库名称: {args.db_name}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 {args.input}")
        exit(1)
    
    # 解析文档
    json_content, md_content = parse_document(args.input, args.output, args.method)
    
    if json_content is not None:
        print("\n✅ 文档解析成功!")
        
        # 加载解析后的内容
        content = load_parsed_content(args.output)
        
        if content:
            print(f"✅ 内容加载成功!")
            print(f"   - JSON数据路径: {content.get('json_path', 'N/A')}")
            print(f"   - MD数据路径: {content.get('md_path', 'N/A')}")
            
            if args.db_name:
                print(f"   - 数据库名称: {args.db_name}")
        else:
            print("⚠️  未找到解析后的内容文件")
    else:
        print("❌ 文档解析失败!")
        exit(1)
    
    print(f"\nStep 1 完成! 接下来运行 Step 2 构建向量数据库")
    if args.db_name:
        print(f"建议命令: python step2_vector_database.py --db-name {args.db_name}") 