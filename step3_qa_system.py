#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: 问答系统前后端
基于Streamlit构建的RAG问答系统，支持多轮对话和上下文记忆
"""

import os
import streamlit as st
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any
import json
from datetime import datetime

# 加载环境变量
load_dotenv()

class RAGChatbot:
    """RAG聊天机器人类"""
    
    def __init__(self, vector_db_path: str = "./vector_db", 
                 chat_model: str = "gpt-4.1-2025-04-14",
                 embedding_model: str = "text-embedding-3-large"):
        """
        初始化RAG聊天机器人
        
        Args:
            vector_db_path (str): 向量数据库路径
            chat_model (str): 对话模型名称
            embedding_model (str): 嵌入模型名称
        """
        self.vector_db_path = vector_db_path
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        
        # 初始化OpenAI客户端
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请在.env文件中设置OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # 加载向量数据库
        self.load_vector_database()
        
        # 初始化对话历史
        self.conversation_history = []
    
    def load_vector_database(self):
        """加载向量数据库"""
        try:
            # 加载FAISS索引
            index_path = os.path.join(self.vector_db_path, "index.faiss")
            print("index_path:",index_path)
            self.index = faiss.read_index(index_path)
            
            # 加载文本和元数据
            with open(os.path.join(self.vector_db_path, "texts.pkl"), 'rb') as f:
                self.texts = pickle.load(f)
            
            with open(os.path.join(self.vector_db_path, "metadata.pkl"), 'rb') as f:
                self.metadata = pickle.load(f)
            
            # 加载配置
            with open(os.path.join(self.vector_db_path, "config.json"), 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            print(f"✅ 向量数据库加载成功: {len(self.texts)} 个文档")
            
        except Exception as e:
            raise Exception(f"加载向量数据库失败: {str(e)}")
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示"""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索相似文档
        
        Args:
            query (str): 查询文本
            top_k (int): 返回文档数量
            
        Returns:
            List[Dict]: 相似文档列表
        """
        # 获取查询向量
        query_embedding = self.get_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # 标准化查询向量
        faiss.normalize_L2(query_vector)
        
        # 搜索相似文档
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # 有效索引
                result = {
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(distance),
                    'rank': i + 1
                }
                results.append(result)
        
        return results
    
    def generate_context_from_documents(self, documents: List[Dict]) -> str:
        """从检索到的文档生成上下文"""
        context_parts = []
        
        for doc in documents:
            source_info = ""
            if doc['metadata'].get('source') == 'markdown':
                section_title = doc['metadata'].get('section_title', 'N/A')
                section_level = doc['metadata'].get('section_level', 0)
                source_info = f"[文档章节 - {section_title} (Level {section_level})]"
            
            context_parts.append(f"{source_info}\n{doc['text']}\n")
        
        return "\n---\n".join(context_parts)
    
    def generate_response(self, user_query: str, context: str, 
                         conversation_history: List[Dict] = None) -> str:
        """
        生成回答
        
        Args:
            user_query (str): 用户问题
            context (str): 检索到的上下文
            conversation_history (List[Dict]): 对话历史
            
        Returns:
            str: 生成的回答
        """
        # 构建系统提示
        system_prompt = """你是一个智能助手，专门帮助企业内部提效。你会基于提供的文档内容来回答问题。

请遵循以下规则：
1. 主要基于提供的上下文内容来回答问题
2. 如果上下文中没有相关信息，请诚实地说明
3. 保持回答的准确性和专业性  
4. 如果可能，提供具体的引用或出处
5. 回答要简洁明了，便于理解
6. 支持中文对话

上下文信息：
{context}
"""
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt.format(context=context)}
        ]
        
        # 添加对话历史（保留最近5轮对话）
        if conversation_history:
            recent_history = conversation_history[-10:]  # 最近10条消息（5轮对话）
            for msg in recent_history:
                messages.append(msg)
        
        # 添加当前用户问题
        messages.append({"role": "user", "content": user_query})
        
        # 生成回答
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.7,
                # max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"生成回答时出错: {str(e)}"
    
    def chat(self, user_query: str) -> Dict[str, Any]:
        """
        主要的聊天方法
        
        Args:
            user_query (str): 用户问题
            
        Returns:
            Dict[str, Any]: 包含回答和相关信息的字典
        """
        # 搜索相关文档
        similar_docs = self.search_similar_documents(user_query, top_k=5)
        
        # 生成上下文
        context = self.generate_context_from_documents(similar_docs)
        
        # 生成回答
        response = self.generate_response(user_query, context, self.conversation_history)
        
        # 更新对话历史
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return {
            'response': response,
            'context': context,
            'similar_documents': similar_docs,
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []

def get_available_databases():
    """获取可用的数据库列表"""
    databases = []
    
    # 默认数据库
    if os.path.exists("./vector_db"):
        databases.append({"name": "默认数据库", "path": "./vector_db"})
    
    # 扫描命名数据库
    for item in os.listdir("."):
        if item.startswith("vector_db_") and os.path.isdir(item):
            db_name = item.replace("vector_db_", "")
            databases.append({"name": db_name, "path": item})
    
    return databases

def init_streamlit_ui():
    """初始化Streamlit界面"""
    st.set_page_config(
        page_title="母线相关知识问答Agent系统",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 母线相关知识问答Agent系统")
    st.markdown("### 企业内部提效的智能问答助手")
    
    # 侧边栏
    with st.sidebar:
        st.header("🗄️ 数据库选择")
        
        # 获取可用数据库
        available_dbs = get_available_databases()
        
        if not available_dbs:
            st.error("❌ 未找到向量数据库!")
            st.info("请先运行 Step 1 和 Step 2 创建数据库")
            st.stop()
        
        # 数据库选择器
        db_options = [f"{db['name']} ({db['path']})" for db in available_dbs]
        selected_db_index = st.selectbox(
            "选择数据库",
            range(len(db_options)),
            format_func=lambda x: db_options[x],
            help="选择要使用的向量数据库"
        )
        
        selected_db = available_dbs[selected_db_index]
        selected_db_path = selected_db["path"]
        
        # 显示数据库信息
        if os.path.exists(os.path.join(selected_db_path, "config.json")):
            try:
                with open(os.path.join(selected_db_path, "config.json"), 'r') as f:
                    config = json.load(f)
                st.info(f"📊 数据库信息:\n- 向量数: {config.get('total_vectors', 'N/A')}\n- 维度: {config.get('dimension', 'N/A')}")
            except:
                pass
        
        st.divider()
        
        st.header("⚙️ 系统配置")
        
        # 模型选择
        chat_model = st.selectbox(
            "对话模型",
            ["gpt-4.1-2025-04-14"],
            index=0
        )
        
        # 检索设置
        top_k = st.slider("检索文档数量", 1, 10, 5)
        
        # 清除历史按钮
        if st.button("🗑️ 清除对话历史"):
            if 'chatbot' in st.session_state:
                st.session_state.chatbot.clear_history()
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.success("对话历史已清除!")
            st.rerun()
        
        # 系统状态
        st.header("📊 系统状态")
        if 'chatbot' in st.session_state:
            chatbot = st.session_state.chatbot
            st.metric("当前数据库", selected_db["name"])
            st.metric("索引文档数", len(chatbot.texts))
            st.metric("对话轮数", len(chatbot.conversation_history) // 2)
    
    return chat_model, top_k, selected_db_path

def main():
    """主函数"""
    # 初始化UI
    chat_model, top_k, selected_db_path = init_streamlit_ui()
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ 请在.env文件中设置OPENAI_API_KEY")
        st.stop()
    
    # 检查向量数据库
    if not os.path.exists(selected_db_path):
        st.error(f"❌ 未找到向量数据库: {selected_db_path}")
        st.stop()
    
    # 数据库路径或模型变化时重新初始化聊天机器人
    chatbot_key = f"chatbot_{selected_db_path}_{chat_model}"
    
    if chatbot_key not in st.session_state or st.session_state.get('current_db_path') != selected_db_path:
        try:
            with st.spinner("🔄 正在加载向量数据库..."):
                st.session_state[chatbot_key] = RAGChatbot(
                    vector_db_path=selected_db_path,
                    chat_model=chat_model
                )
                st.session_state.chatbot = st.session_state[chatbot_key]
                st.session_state.current_db_path = selected_db_path
                
                # 清除消息历史（切换数据库时）
                if 'messages' in st.session_state:
                    st.session_state.messages = []
                    
            st.success(f"✅ 已加载数据库: {selected_db_path}")
        except Exception as e:
            st.error(f"❌ 系统初始化失败: {str(e)}")
            st.stop()
    else:
        # 使用已存在的聊天机器人
        st.session_state.chatbot = st.session_state[chatbot_key]
    
    # 初始化消息历史
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "您好！我是您的智能问答助手,请随时提问！"}
        ]
    
    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 生成并显示助手回答
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在思考..."):
                try:
                    # 获取聊天机器人回答
                    result = st.session_state.chatbot.chat(prompt)
                    response = result['response']
                    
                    # 显示回答
                    st.markdown(response)
                    
                    # 显示相关文档（可展开）
                    with st.expander("📚 相关文档章节", expanded=False):
                        for i, doc in enumerate(result['similar_documents']):
                            st.markdown(f"**章节 {i+1}** (相似度: {doc['similarity']:.3f})")
                            
                            # 显示章节标题
                            metadata = doc['metadata']
                            if metadata.get('source') == 'markdown':
                                section_title = metadata.get('section_title', 'N/A')
                                section_level = metadata.get('section_level', 0)
                                st.markdown(f"**章节标题**: {section_title}")
                                st.caption(f"章节级别: Level {section_level}")
                            
                            # 显示文本内容
                            text_preview = doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text']
                            st.markdown(f"```\n{text_preview}\n```")
                            
                            st.divider()
                    
                except Exception as e:
                    response = f"抱歉，处理您的问题时出现错误：{str(e)}"
                    st.error(response)
        
        # 添加助手消息到历史
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 