# RAG问答Agent系统

企业内部提效的智能问答系统，基于RAG（Retrieval-Augmented Generation）技术。

## 功能特点

- PDF文档解析和内容提取
- 基于Markdown章节的智能分割
- **多数据库支持** - 为不同领域创建独立的向量数据库
- 向量数据库索引和检索
- 多轮对话支持
- 上下文记忆功能
- 数据库管理工具

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

1. 在 `.env` 文件中配置您的 OpenAI API 密钥：
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 使用方法

### 🚀 快速开始 (默认数据库)

```bash
# 一键运行完整流程
python run_rag_system.py --all

# 或分步运行
python run_rag_system.py --step 1    # 文档处理
python run_rag_system.py --step 2    # 构建向量数据库
python run_rag_system.py --step 3    # 启动问答系统
```

### 🗄️ 多数据库支持

为不同领域创建独立的数据库：

```bash
# 创建金融领域数据库
python run_rag_system.py --all -d finance -i ./docs/finance.pdf

# 创建医疗领域数据库  
python run_rag_system.py --all -d medical -i ./docs/medical.pdf

# 启动问答系统 (在Web界面中选择数据库)
python run_rag_system.py --step 3
```

### 🛠️ 数据库管理

```bash
# 交互式数据库管理
python manage_databases.py

# 列出所有数据库
python manage_databases.py list

# 创建新数据库 (向导模式)
python manage_databases.py create
```

### 📝 手动运行

```bash
# Step 1: 文档处理
python step1_document_parser.py --input file.pdf --db-name mydb

# Step 2: 向量数据库构建
python step2_vector_database.py --db-name mydb

# Step 3: 启动问答系统
streamlit run step3_qa_system.py
```

⚠️ **注意**: 请使用 `streamlit run` 命令启动问答系统，不要直接用 `python` 运行。

## 项目结构

### 📁 核心目录
- `input_files/`: 输入的PDF文档
- `output/`: 默认数据库的解析内容
- `output_[数据库名]/`: 命名数据库的解析内容
- `vector_db/`: 默认向量数据库
- `vector_db_[数据库名]/`: 命名向量数据库
- `backups/`: 数据库备份文件

### 🐍 核心脚本
- `step1_document_parser.py`: 文档解析模块
- `step2_vector_database.py`: 向量数据库构建模块
- `step3_qa_system.py`: 问答系统前端
- `run_rag_system.py`: 一键运行脚本
- `manage_databases.py`: 数据库管理工具
- `diagnose_system.py`: 系统诊断工具
- `test_md_parsing.py`: MD解析测试脚本 