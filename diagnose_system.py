#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统诊断脚本 - 检查RAG系统的各个组件是否正常
"""

import os
import sys

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查Python依赖包...")
    
    required_packages = [
        'streamlit',
        'openai', 
        'faiss',
        'numpy',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_env_file():
    """检查环境变量文件"""
    print("\n🔍 检查环境配置...")
    
    if not os.path.exists('.env'):
        print("  ❌ .env文件不存在")
        print("  请创建.env文件并添加: OPENAI_API_KEY=your_key_here")
        return False
    
    print("  ✅ .env文件存在")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("  ❌ OPENAI_API_KEY未设置")
            return False
        elif api_key == "your_openai_api_key_here":
            print("  ❌ OPENAI_API_KEY是默认值，请设置真实的API密钥")
            return False
        else:
            print("  ✅ OPENAI_API_KEY已设置")
            return True
            
    except Exception as e:
        print(f"  ❌ 读取.env文件失败: {str(e)}")
        return False

def check_files():
    """检查必要文件"""
    print("\n🔍 检查项目文件...")
    
    required_files = [
        'step1_document_parser.py',
        'step2_vector_database.py', 
        'step3_qa_system.py',
        'input_files/upfluencer.pdf'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_exist = False
    
    return all_exist

def check_processed_data():
    """检查处理后的数据"""
    print("\n🔍 检查处理后的数据...")
    
    # 检查step1输出
    output_found = False
    if os.path.exists('./output'):
        for root, dirs, files in os.walk('./output'):
            for file in files:
                if file.endswith('_content_list.json') or file.endswith('.md'):
                    output_found = True
                    print(f"  ✅ 找到输出文件: {os.path.join(root, file)}")
    
    if not output_found:
        print("  ❌ 未找到Step1输出文件，请先运行: python step1_document_parser.py")
        return False
    
    # 检查step2输出
    vector_db_files = ['index.faiss', 'texts.pkl', 'metadata.pkl', 'config.json']
    vector_db_found = True
    
    for file in vector_db_files:
        file_path = f'./vector_db/{file}'
        if os.path.exists(file_path):
            print(f"  ✅ 向量数据库文件: {file}")
        else:
            print(f"  ❌ 向量数据库文件: {file}")
            vector_db_found = False
    
    if not vector_db_found:
        print("  ❌ 向量数据库不完整，请先运行: python step2_vector_database.py")
        return False
    
    return True

def test_streamlit_import():
    """测试Streamlit导入"""
    print("\n🔍 测试Streamlit导入...")
    
    try:
        import streamlit as st
        print("  ✅ Streamlit导入成功")
        
        # 测试basic functions
        st.write  # 测试是否可以访问
        st.title
        st.chat_message
        print("  ✅ Streamlit基本功能可用")
        return True
        
    except Exception as e:
        print(f"  ❌ Streamlit导入失败: {str(e)}")
        return False

def test_openai_connection():
    """测试OpenAI连接"""
    print("\n🔍 测试OpenAI连接...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key or api_key == "your_openai_api_key_here":
            print("  ❌ API密钥未正确设置")
            return False
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # 简单测试API连接（不实际调用，只测试client创建）
        print("  ✅ OpenAI客户端创建成功")
        print("  💡 API连接测试跳过（避免费用），实际使用时会测试")
        return True
        
    except Exception as e:
        print(f"  ❌ OpenAI连接测试失败: {str(e)}")
        return False

def main():
    """主诊断函数"""
    print("🤖 RAG问答系统诊断工具")
    print("=" * 50)
    
    checks = [
        ("依赖包检查", check_dependencies),
        ("环境配置检查", check_env_file), 
        ("项目文件检查", check_files),
        ("数据文件检查", check_processed_data),
        ("Streamlit测试", test_streamlit_import),
        ("OpenAI连接测试", test_openai_connection)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"  ❌ {check_name}出错: {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("✅ 所有检查通过！系统应该可以正常运行")
        print("\n推荐运行顺序:")
        print("1. streamlit run step3_qa_system.py")
        print("2. 在浏览器中打开 http://localhost:8501")
    else:
        print("❌ 发现问题，请根据上述检查结果修复")
        print("\n建议修复步骤:")
        print("1. 安装缺少的依赖: pip install -r requirements.txt")
        print("2. 配置API密钥: 编辑.env文件")
        print("3. 运行数据处理: python step1_document_parser.py")
        print("4. 构建向量数据库: python step2_vector_database.py")
        print("5. 启动问答系统: streamlit run step3_qa_system.py")

if __name__ == "__main__":
    main() 