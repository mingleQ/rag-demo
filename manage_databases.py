#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据库管理脚本
帮助用户管理不同领域的RAG向量数据库
"""

import os
import json
import argparse
import shutil
from datetime import datetime

def list_databases():
    """列出所有可用的数据库"""
    print("🗄️ 可用的向量数据库:")
    print("-" * 50)
    
    databases = []
    
    # 默认数据库
    if os.path.exists("./vector_db"):
        databases.append({
            "name": "默认数据库",
            "path": "./vector_db",
            "type": "default"
        })
    
    # 命名数据库
    for item in os.listdir("."):
        if item.startswith("vector_db_") and os.path.isdir(item):
            db_name = item.replace("vector_db_", "")
            databases.append({
                "name": db_name,
                "path": item,
                "type": "named"
            })
    
    if not databases:
        print("❌ 未找到任何数据库")
        print("💡 请先运行 Step 1 和 Step 2 创建数据库")
        return
    
    for i, db in enumerate(databases, 1):
        print(f"{i}. {db['name']}")
        print(f"   路径: {db['path']}")
        
        # 显示数据库统计信息
        config_path = os.path.join(db['path'], 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"   向量数: {config.get('total_vectors', 'N/A')}")
                print(f"   维度: {config.get('dimension', 'N/A')}")
                print(f"   模型: {config.get('model_name', 'N/A')}")
            except:
                print("   状态: 配置文件读取失败")
        else:
            print("   状态: 未完成构建")
        
        # 检查源文件
        if db['type'] == 'named':
            output_dir = f"./output_{db['name']}"
            if os.path.exists(output_dir):
                print(f"   源文件: ✅ {output_dir}")
            else:
                print(f"   源文件: ❌ {output_dir}")
        else:
            if os.path.exists("./output"):
                print(f"   源文件: ✅ ./output")
            else:
                print(f"   源文件: ❌ ./output")
        
        print()

def create_database():
    """创建新数据库的向导"""
    print("🆕 创建新数据库向导")
    print("-" * 50)
    
    # 获取数据库名称
    db_name = input("请输入数据库名称 (英文，用于区分不同领域): ").strip()
    if not db_name:
        print("❌ 数据库名称不能为空")
        return
    
    if not db_name.replace('_', '').replace('-', '').isalnum():
        print("❌ 数据库名称只能包含字母、数字、下划线和连字符")
        return
    
    # 检查是否已存在
    db_path = f"./vector_db_{db_name}"
    if os.path.exists(db_path):
        print(f"❌ 数据库 '{db_name}' 已存在")
        return
    
    # 获取PDF文件路径
    pdf_path = input("请输入PDF文件路径: ").strip()
    if not pdf_path:
        print("❌ PDF文件路径不能为空")
        return
    
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        return
    
    if not pdf_path.lower().endswith('.pdf'):
        print("❌ 请提供PDF文件")
        return
    
    print(f"\n📋 创建数据库配置:")
    print(f"   数据库名称: {db_name}")
    print(f"   PDF文件: {pdf_path}")
    print(f"   输出目录: ./output_{db_name}")
    print(f"   数据库目录: ./vector_db_{db_name}")
    
    confirm = input("\n确认创建? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 取消创建")
        return
    
    # 运行Step 1
    print("\n🔄 Step 1: 解析PDF文档...")
    import subprocess
    import sys
    
    cmd1 = [
        sys.executable, "step1_document_parser.py",
        "--input", pdf_path,
        "--db-name", db_name
    ]
    
    try:
        result = subprocess.run(cmd1, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print(f"❌ Step 1 失败: {result.stderr}")
            return
        print("✅ Step 1 完成")
    except Exception as e:
        print(f"❌ Step 1 执行失败: {str(e)}")
        return
    
    # 运行Step 2
    print("\n🔄 Step 2: 构建向量数据库...")
    cmd2 = [
        sys.executable, "step2_vector_database.py",
        "--db-name", db_name
    ]
    
    try:
        result = subprocess.run(cmd2, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print(f"❌ Step 2 失败: {result.stderr}")
            return
        print("✅ Step 2 完成")
    except Exception as e:
        print(f"❌ Step 2 执行失败: {str(e)}")
        return
    
    print(f"\n🎉 数据库 '{db_name}' 创建成功!")
    print("现在可以在Step 3中选择使用这个数据库")

def delete_database():
    """删除数据库"""
    print("🗑️ 删除数据库")
    print("-" * 50)
    
    # 列出可删除的数据库（不包括默认数据库）
    databases = []
    for item in os.listdir("."):
        if item.startswith("vector_db_") and os.path.isdir(item):
            db_name = item.replace("vector_db_", "")
            databases.append({"name": db_name, "path": item})
    
    if not databases:
        print("❌ 没有可删除的命名数据库")
        return
    
    print("可删除的数据库:")
    for i, db in enumerate(databases, 1):
        print(f"{i}. {db['name']} ({db['path']})")
    
    try:
        choice = int(input("\n请选择要删除的数据库编号: ")) - 1
        if choice < 0 or choice >= len(databases):
            print("❌ 无效选择")
            return
        
        selected_db = databases[choice]
        
        print(f"\n⚠️ 即将删除数据库: {selected_db['name']}")
        print(f"   数据库路径: {selected_db['path']}")
        print(f"   源文件路径: ./output_{selected_db['name']}")
        
        confirm = input("确认删除? 此操作不可恢复! (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 取消删除")
            return
        
        # 删除数据库目录
        if os.path.exists(selected_db['path']):
            shutil.rmtree(selected_db['path'])
            print(f"✅ 已删除数据库目录: {selected_db['path']}")
        
        # 删除源文件目录
        output_dir = f"./output_{selected_db['name']}"
        if os.path.exists(output_dir):
            delete_output = input(f"是否同时删除源文件 {output_dir}? (y/N): ").strip().lower()
            if delete_output == 'y':
                shutil.rmtree(output_dir)
                print(f"✅ 已删除源文件目录: {output_dir}")
        
        print(f"🎉 数据库 '{selected_db['name']}' 删除成功!")
        
    except ValueError:
        print("❌ 请输入有效的数字")
    except Exception as e:
        print(f"❌ 删除失败: {str(e)}")

def backup_database():
    """备份数据库"""
    print("💾 备份数据库")
    print("-" * 50)
    
    # 列出所有数据库
    databases = []
    if os.path.exists("./vector_db"):
        databases.append({"name": "默认数据库", "path": "./vector_db", "type": "default"})
    
    for item in os.listdir("."):
        if item.startswith("vector_db_") and os.path.isdir(item):
            db_name = item.replace("vector_db_", "")
            databases.append({"name": db_name, "path": item, "type": "named"})
    
    if not databases:
        print("❌ 没有可备份的数据库")
        return
    
    print("可备份的数据库:")
    for i, db in enumerate(databases, 1):
        print(f"{i}. {db['name']} ({db['path']})")
    
    try:
        choice = int(input("\n请选择要备份的数据库编号: ")) - 1
        if choice < 0 or choice >= len(databases):
            print("❌ 无效选择")
            return
        
        selected_db = databases[choice]
        
        # 创建备份目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"./backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_name = f"{selected_db['name']}_{timestamp}"
        backup_path = os.path.join(backup_dir, backup_name)
        
        # 复制数据库
        shutil.copytree(selected_db['path'], backup_path)
        
        print(f"✅ 数据库备份成功!")
        print(f"   备份路径: {backup_path}")
        
    except ValueError:
        print("❌ 请输入有效的数字")
    except Exception as e:
        print(f"❌ 备份失败: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RAG数据库管理工具')
    parser.add_argument('action', nargs='?', choices=['list', 'create', 'delete', 'backup'],
                       help='操作类型')
    
    args = parser.parse_args()
    
    print("🤖 RAG数据库管理工具")
    print("=" * 50)
    
    if args.action == 'list':
        list_databases()
    elif args.action == 'create':
        create_database()
    elif args.action == 'delete':
        delete_database()
    elif args.action == 'backup':
        backup_database()
    else:
        # 交互式菜单
        while True:
            print("\n请选择操作:")
            print("1. 📋 列出所有数据库")
            print("2. 🆕 创建新数据库")
            print("3. 🗑️ 删除数据库")
            print("4. 💾 备份数据库")
            print("5. 🚪 退出")
            
            try:
                choice = input("\n请输入选择 (1-5): ").strip()
                
                if choice == '1':
                    list_databases()
                elif choice == '2':
                    create_database()
                elif choice == '3':
                    delete_database()
                elif choice == '4':
                    backup_database()
                elif choice == '5':
                    print("👋 再见!")
                    break
                else:
                    print("❌ 无效选择，请输入1-5")
                    
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 操作失败: {str(e)}")

if __name__ == "__main__":
    main() 