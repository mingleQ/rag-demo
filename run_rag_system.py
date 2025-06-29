#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG问答系统一键运行脚本
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import raganything
        import faiss
        import openai
        import streamlit
        import langchain
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_env_file():
    """检查.env文件"""
    if not os.path.exists('.env'):
        print("❌ 未找到.env文件")
        print("请创建.env文件并添加以下内容:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ 请在.env文件中设置有效的OPENAI_API_KEY")
        return False
    
    return True

def run_step1(db_name=None, input_pdf=None):
    """运行Step 1: 文档处理"""
    print("\n" + "="*50)
    print("Step 1: 开始文档处理...")
    print("="*50)
    
    try:
        cmd = [sys.executable, "step1_document_parser.py"]
        
        if db_name:
            cmd.extend(["--db-name", db_name])
        
        if input_pdf:
            cmd.extend(["--input", input_pdf])
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
        
        if result.returncode != 0:
            print("❌ Step 1 执行失败!")
            return False
        
        print("✅ Step 1 完成!")
        return True
        
    except Exception as e:
        print(f"❌ Step 1 执行出错: {str(e)}")
        return False

def run_step2(db_name=None):
    """运行Step 2: 向量数据库构建"""
    print("\n" + "="*50)
    print("Step 2: 开始构建向量数据库...")
    print("="*50)
    
    try:
        cmd = [sys.executable, "step2_vector_database.py"]
        
        if db_name:
            cmd.extend(["--db-name", db_name])
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
        
        if result.returncode != 0:
            print("❌ Step 2 执行失败!")
            return False
        
        print("✅ Step 2 完成!")
        return True
        
    except Exception as e:
        print(f"❌ Step 2 执行出错: {str(e)}")
        return False

def run_step3():
    """运行Step 3: 启动问答系统"""
    print("\n" + "="*50)
    print("Step 3: 启动问答系统...")
    print("="*50)
    
    try:
        print("🚀 正在启动Streamlit应用...")
        print("请在浏览器中打开显示的URL")
        print("按 Ctrl+C 停止服务")
        
        # 启动Streamlit应用
        subprocess.run([sys.executable, "-m", "streamlit", "run", "step3_qa_system.py"])
        
    except KeyboardInterrupt:
        print("\n👋 系统已停止")
    except Exception as e:
        print(f"❌ Step 3 执行出错: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RAG问答系统运行脚本')
    parser.add_argument('--step', type=int, choices=[1, 2, 3], 
                       help='运行指定步骤 (1: 文档处理, 2: 向量数据库, 3: 问答系统)')
    parser.add_argument('--all', action='store_true', 
                       help='运行所有步骤')
    parser.add_argument('--skip-check', action='store_true',
                       help='跳过依赖检查')
    parser.add_argument('--db-name', '-d',
                       help='数据库名称 (用于区分不同领域的文档)')
    parser.add_argument('--input-pdf', '-i',
                       help='PDF文件路径 (与--db-name一起使用)')
    
    args = parser.parse_args()
    
    print("🤖 RAG问答Agent系统")
    print("企业内部提效的智能问答助手")
    print("-" * 50)
    
    # 检查依赖
    if not args.skip_check:
        print("检查系统依赖...")
        if not check_dependencies():
            return
        
        if not check_env_file():
            return
        
        print("✅ 系统检查通过!")
    
    # 检查输入文件
    if not os.path.exists("./input_files/upfluencer.pdf"):
        print("❌ 未找到输入文件: ./input_files/upfluencer.pdf")
        return
    
    # 检查多数据库参数
    if args.db_name and args.step == 3:
        print("❌ Step 3 不需要指定数据库名称，请在Web界面中选择")
        return
    
    # 运行指定步骤
    if args.step == 1:
        run_step1(args.db_name, args.input_pdf)
    elif args.step == 2:
        # 检查输入目录
        input_dir = f"./output_{args.db_name}" if args.db_name else "./output"
        if not os.path.exists(input_dir):
            print(f"❌ 未找到输出目录: {input_dir}")
            print("请先运行Step 1")
            return
        run_step2(args.db_name)
    elif args.step == 3:
        # 检查是否有任何数据库
        has_db = os.path.exists("./vector_db")
        if not has_db:
            for item in os.listdir("."):
                if item.startswith("vector_db_") and os.path.isdir(item):
                    has_db = True
                    break
        
        if not has_db:
            print("❌ 未找到任何向量数据库，请先运行Step 1和Step 2")
            return
        run_step3()
    elif args.all:
        # 运行所有步骤
        if args.db_name:
            print(f"🚀 开始为数据库 '{args.db_name}' 运行完整流程...")
        else:
            print("🚀 开始运行完整流程...")
        
        # Step 1
        if not run_step1(args.db_name, args.input_pdf):
            return
        
        # Step 2
        if not run_step2(args.db_name):
            return
        
        # Step 3
        print("\n所有步骤完成! 现在启动问答系统...")
        input("按Enter键继续启动问答系统...")
        run_step3()
    else:
        # 显示帮助信息
        print("请使用以下命令运行系统:")
        print("\n📋 基本用法:")
        print("  python run_rag_system.py --all                    # 运行完整流程(默认数据库)")
        print("  python run_rag_system.py --step 1                 # 只运行文档处理")
        print("  python run_rag_system.py --step 2                 # 只运行向量数据库构建")
        print("  python run_rag_system.py --step 3                 # 只运行问答系统")
        
        print("\n🗄️ 多数据库用法:")
        print("  python run_rag_system.py --all -d finance -i file.pdf    # 创建finance数据库")
        print("  python run_rag_system.py --step 1 -d medical -i doc.pdf  # 为medical领域处理文档")
        print("  python run_rag_system.py --step 2 -d medical             # 构建medical数据库")
        
        print("\n🛠️ 数据库管理:")
        print("  python manage_databases.py                        # 交互式数据库管理")
        print("  python manage_databases.py list                   # 列出所有数据库")
        print("  python manage_databases.py create                 # 创建新数据库")
        
        print("\n📝 手动运行:")
        print("  python step1_document_parser.py --help")
        print("  python step2_vector_database.py --help")
        print("  streamlit run step3_qa_system.py")

if __name__ == "__main__":
    main() 