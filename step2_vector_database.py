#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: 向量数据库构建模块
使用OpenAI embedding模型和FAISS构建向量索引
"""

import os
import json
import pickle
import numpy as np
from dotenv import load_dotenv
import faiss
from openai import OpenAI
from typing import List, Dict, Any
import time
import hashlib

# 加载环境变量
load_dotenv()

class VectorDatabase:
    """向量数据库类"""
    
    def __init__(self, api_key=None, model_name="text-embedding-3-small"):
        """
        初始化向量数据库
        
        Args:
            api_key (str): OpenAI API密钥
            model_name (str): embedding模型名称
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
        self.index = None
        self.texts = []
        self.metadata = []
        self.embeddings = []
        
        if not self.api_key:
            raise ValueError("请在.env文件中设置OPENAI_API_KEY")
    
    def get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """
        获取文本的向量表示
        
        Args:
            text (str): 输入文本
            max_retries (int): 最大重试次数
            
        Returns:
            List[float]: 向量表示
        """
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"获取embedding失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise e
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        将文本分块
        
        Args:
            text (str): 输入文本
            chunk_size (int): 块大小
            overlap (int): 重叠大小
            
        Returns:
            List[str]: 文本块列表
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
                
            start = end - overlap
        
        return chunks
    
    def process_md_content(self, md_text: str, file_path: str = '', min_chunk_size: int = 2000) -> List[Dict]:
        """
        处理MD内容，使用#号分割章节，但合并较短的章节以达到最小长度要求
        
        Args:
            md_text (str): MD文本内容
            file_path (str): 文件路径
            min_chunk_size (int): 最小块大小，默认2000字符
            
        Returns:
            List[Dict]: 处理后的文档列表
        """
        # 首先按章节分割，获取所有章节
        raw_sections = []
        lines = md_text.split('\n')
        current_section = []
        current_title = ""
        section_level = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # 检查是否是标题行
            if line_stripped.startswith('#'):
                # 如果之前有内容，保存之前的章节
                if current_section and current_title:
                    content = '\n'.join(current_section).strip()
                    if content:  # 跳过空章节
                        raw_sections.append({
                            'text': content,
                            'title': current_title,
                            'level': section_level
                        })
                
                # 开始新章节
                current_title = line_stripped
                section_level = len(line_stripped) - len(line_stripped.lstrip('#'))
                current_section = [line]  # 包含标题（保持原始格式）
                
            else:
                # 添加到当前章节
                current_section.append(line)  # 保持原始格式
        
        # 处理最后一个章节
        if current_section and current_title:
            content = '\n'.join(current_section).strip()
            if content:  # 跳过空章节
                raw_sections.append({
                    'text': content,
                    'title': current_title,
                    'level': section_level
                })
        
        if not raw_sections:
            return []
        
        # 合并较短的章节
        documents = []
        current_chunk = ""
        current_titles = []
        current_levels = []
        
        for i, section in enumerate(raw_sections):
            # 如果当前块为空，直接添加这个章节
            if not current_chunk:
                current_chunk = section['text']
                current_titles = [section['title']]
                current_levels = [section['level']]
            else:
                # 检查添加这个章节后是否会超过合理长度
                potential_chunk = current_chunk + '\n\n' + section['text']
                
                # 如果当前块已经足够长，或者添加下一个章节会使块过长，则完成当前块
                if len(current_chunk) >= min_chunk_size or len(potential_chunk) > min_chunk_size * 2:
                    # 保存当前块
                    main_title = current_titles[0]  # 使用第一个章节的标题作为主标题
                    combined_titles = " | ".join(current_titles) if len(current_titles) > 1 else main_title
                    
                    doc = {
                        'text': current_chunk,
                        'source': 'markdown',
                        'title': main_title,
                        'combined_titles': combined_titles,
                        'level': min(current_levels),  # 使用最高级别（数字最小）
                        'metadata': {
                            'file_path': file_path,
                            'section_title': main_title,
                            'combined_section_titles': combined_titles,
                            'section_level': min(current_levels),
                            'sections_count': len(current_titles),
                            'char_count': len(current_chunk)
                        }
                    }
                    documents.append(doc)
                    
                    # 开始新块
                    current_chunk = section['text']
                    current_titles = [section['title']]
                    current_levels = [section['level']]
                else:
                    # 合并到当前块
                    current_chunk = potential_chunk
                    current_titles.append(section['title'])
                    current_levels.append(section['level'])
        
        # 处理最后一个块
        if current_chunk:
            main_title = current_titles[0]
            combined_titles = " | ".join(current_titles) if len(current_titles) > 1 else main_title
            
            doc = {
                'text': current_chunk,
                'source': 'markdown',
                'title': main_title,
                'combined_titles': combined_titles,
                'level': min(current_levels),
                'metadata': {
                    'file_path': file_path,
                    'section_title': main_title,
                    'combined_section_titles': combined_titles,
                    'section_level': min(current_levels),
                    'sections_count': len(current_titles),
                    'char_count': len(current_chunk)
                }
            }
            documents.append(doc)
        
        return documents
    
    def build_index(self, content_list) -> bool:
        """
        构建向量索引
        
        Args:
            content_data (List): 包含MD数据的字典
            
        Returns:
            bool: 是否成功
        """
        print("开始构建向量索引...")
        
        documents = []

        for content_data in content_list:
        
            # 只处理MD数据，使用#号分割章节
            if 'md_data' in content_data and content_data['md_data']:
                print(f"处理MD数据: {len(content_data['md_data'])} 个字符")
                md_text = content_data['md_data']
                file_path = content_data.get('md_path', '')
                
                # 使用新的MD处理方法
                md_docs = self.process_md_content(md_text, file_path)
                documents.extend(md_docs)
                
                print(f"从MD文件中提取了 {len(md_docs)} 个章节")
            
            # if not documents:
            #     print("❌ 没有找到可处理的文档内容")
            #     return False
        
        print(f"总共处理 {len(documents)} 个文档章节")
        
        # 生成embeddings
        print("生成向量表示...")
        embeddings = []
        texts = []
        metadata = []
        
        for i, doc in enumerate(documents):
            try:
                print(f"处理第 {i+1}/{len(documents)} 个文档块...")
                embedding = self.get_embedding(doc['text'])
                embeddings.append(embedding)
                texts.append(doc['text'])
                metadata.append(doc)
                
                # 添加小延迟避免API限制
                time.sleep(0.1)
                
            except Exception as e:
                print(f"处理文档块 {i} 失败: {str(e)}")
                continue
        
        if not embeddings:
            print("❌ 没有成功生成任何向量表示")
            return False
        
        print(f"成功生成 {len(embeddings)} 个向量表示")
        
        # 构建FAISS索引
        print("构建FAISS索引...")
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        # 创建FAISS索引（使用余弦相似度）
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # 标准化向量以使用余弦相似度
        faiss.normalize_L2(embeddings_array)
        
        # 添加向量到索引
        self.index.add(embeddings_array)
        
        # 保存数据
        self.texts = texts
        self.metadata = metadata
        self.embeddings = embeddings
        
        print(f"✅ 向量索引构建完成! 索引包含 {self.index.ntotal} 个向量")
        return True
    
    def save_database(self, db_dir: str = "./vector_db"):
        """
        保存向量数据库
        
        Args:
            db_dir (str): 数据库保存目录
        """
        os.makedirs(db_dir, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(db_dir, "index.faiss"))
        
        # 保存文本和元数据
        with open(os.path.join(db_dir, "texts.pkl"), 'wb') as f:
            pickle.dump(self.texts, f)
        
        with open(os.path.join(db_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # 保存配置信息
        config = {
            'model_name': self.model_name,
            'dimension': self.index.d if self.index else 0,
            'total_vectors': self.index.ntotal if self.index else 0
        }
        
        with open(os.path.join(db_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 向量数据库已保存到: {db_dir}")
    
    def load_database(self, db_dir: str = "./vector_db"):
        """
        加载向量数据库
        
        Args:
            db_dir (str): 数据库目录
        """
        # 加载FAISS索引
        index_path = os.path.join(db_dir, "index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # 加载文本和元数据
        texts_path = os.path.join(db_dir, "texts.pkl")
        if os.path.exists(texts_path):
            with open(texts_path, 'rb') as f:
                self.texts = pickle.load(f)
        
        metadata_path = os.path.join(db_dir, "metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        print(f"✅ 向量数据库已加载: {len(self.texts)} 个文档")

def load_parsed_content(output_dir: str = "./output"):
    """
    加载解析后的内容
    
    Args:
        output_dir (str): 输出目录
        
    Returns:
        Dict: 内容数据
    """
    content_list = []
    content = {}
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
                    
            if file.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    print("file_path:",file_path)
                    content['md_data'] = f.read()
                    content['md_path'] = file_path
                    content_list.append(content)
                    content = {}
    
    return content_list

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    # python step2_vector_database.py -i ./output2  -d "vector_db_iec" -m "text-embedding-3-large"

    parser = argparse.ArgumentParser(description='向量数据库构建器')
    parser.add_argument('--input-dir', '-i', default="./output", 
                       help='输入目录 (解析后的内容)')
    parser.add_argument('--db-dir', '-d', default="./vector_db", 
                       help='向量数据库输出目录')
    parser.add_argument('--model', '-m', default="text-embedding-3-small",
                       help='Embedding模型名称')
    
    args = parser.parse_args()
    

    

    
    # 加载解析后的内容
    print("\n加载解析后的内容...")
    content_list = load_parsed_content(args.input_dir)
    
    if len(content_list) == 0:
        print(f"❌ 未找到解析后的内容在 {args.input_dir}")
        print("请先运行 Step 1")
        if args.db_name:
            print(f"建议命令: python step1_document_parser.py --db-name {args.db_name}")
        exit(1)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 创建向量数据库
    try:
        db = VectorDatabase(api_key=api_key, model_name=args.model)
        
        # 构建索引
        success = db.build_index(content_list)
        
        if success:
            # 保存数据库
            db.save_database(args.db_dir)
            print(f"\n✅ 向量数据库构建成功!")
            print(f"数据库路径: {args.db_dir}")
            print("接下来运行 Step 3 启动问答系统")
            # if args.db_name:
            #     print(f"建议命令: streamlit run step3_qa_system.py")
        else:
            print("❌ 向量数据库构建失败!")
            exit(1)
            
    except Exception as e:
        print(f"❌ 构建向量数据库时出错: {str(e)}")
        exit(1) 