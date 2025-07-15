#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import uuid
import requests
from typing import Dict, List, Tuple, Optional
import numpy as np
from py2neo import Graph
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.pretty import Pretty
from rich.markdown import Markdown
from collections import defaultdict  
import argparse
from auth_util import gen_sign_headers

# 定义API配置常量
APP_ID = '2025630384'
APP_KEY = 'fsUlhWWiDgeCqEfi'
DOMAIN = 'api-ai.vivo.com.cn'
LLM_URI = '/vivogpt/completions'
EMBEDDING_URI = '/embedding-model-api/predict/batch'
METHOD = 'POST'



# 初始化rich控制台
console = Console()



class Neo4jRAGSystem:
    BUDGET_MODES = {
        "Deeper": {
            "entity_limit": 3,
            "relation_limit": 10,
            "top_k_triples": 5,
            "one_hop_limit": 10,
            "top_k_multi_hop_entities": 5,
            "multi_hop_limit": 3
        },
        "Deep": {
            "entity_limit": 2,
            "relation_limit": 8,
            "one_hop_limit": 8,
            "top_k_triples": 4,
            "top_k_multi_hop_entities": 4,
            "multi_hop_limit": 2
        }
    }

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 enable_multi_hop: bool = True, search_budget_mode: str = "Deeper"):
        """初始化RAG系统"""
        # 初始化Rich控制台
        self.console = console
        

        self.enable_multi_hop = enable_multi_hop
        
        # 设置搜索预算参数
        if search_budget_mode not in self.BUDGET_MODES:
            self.console.print(f"[bold red]警告：未知的搜索预算模式 '{search_budget_mode}'。将使用默认的 'Deeper' 模式。[/bold red]")
            search_budget_mode = "Deeper"
        self.search_budget = self.BUDGET_MODES[search_budget_mode]
        self.console.print(f"ℹ️ 搜索预算模式已设置为: [bold magenta]{search_budget_mode}[/bold magenta]")
        
        # 显示初始化信息
        with self.console.status("[bold green]正在初始化系统...", spinner="dots"):
            # 初始化Neo4j连接
            self.console.print("🔄 连接Neo4j数据库...", style="blue")
            self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.console.print("✅ Neo4j数据库连接成功", style="green")
            
            # 初始化蓝心大模型客户端
            self.console.print("🔄 初始化蓝心大模型客户端...", style="blue")
            self.console.print("✅ 蓝心大模型客户端初始化成功", style="green")
            
            # 实体类型和关系类型定义
            self.ENTITY_TYPES = [
                'Disease', 'Category', 'Symptom', 'Department', 'Treatment', 
                'Check', 'Drug', 'Food', 'Recipe', 'Person', 'Organization',
                'Time', 'Location', 'Other'
            ]
            
            self.RELATION_TYPES = [
                'BELONGS_TO', 'HAS_SYMPTOM', 'TREATED_BY', 'USES_TREATMENT',
                'REQUIRES_CHECK', 'RECOMMENDS_DRUG', 'COMMONLY_USES_DRUG',
                'SHOULD_EAT', 'SHOULD_NOT_EAT', 'RECOMMENDS_RECIPE', 
                'ACCOMPANIES', 'OTHER'
            ]
            
            # 系统提示词
            self.entity_extraction_prompt = self._get_entity_extraction_prompt()
            self.answer_generation_prompt = self._get_answer_generation_prompt()
            
            self.console.print("✅ 系统初始化完成!", style="bold green")
    
    def _get_entity_extraction_prompt(self) -> str:
        """获取实体抽取的系统提示词"""
        return """你是一个专业的医学实体关系抽取助手。你的任务是从医学问题文本中提取能够解决该问题的关键实体及其关系。

请识别以下类型的实体:
- Disease: 疾病名称，如"肺炎"、"糖尿病"、"高血压"等
- Category: 疾病分类，如"内科"、"呼吸内科"、"心血管内科"等
- Symptom: 疾病症状，如"发热"、"咳嗽"、"胸痛"、"头晕"等  
- Department: 治疗科室，如"内科"、"外科"、"急诊科"等
- Treatment: 治疗方法，如"药物治疗"、"手术治疗"、"康复治疗"等
- Check: 检查项目，如"血常规"、"胸部CT"、"心电图"等
- Drug: 药物名称，如"阿奇霉素"、"青霉素"、"布洛芬"等
- Food: 食物名称，如"鸡蛋"、"牛奶"、"辣椒"等
- Recipe: 推荐食谱，如"百合粥"、"银耳汤"、"蒸蛋羹"等
- Person: 人名、医生、患者等
- Organization: 医院、医疗机构等
- Time: 时间、年龄、病程等
- Location: 地点、部位等
- Other: 其他

请识别以下类型的关系:
- BELONGS_TO: 疾病属于某分类，如"肺炎属于呼吸内科"
- HAS_SYMPTOM: 疾病有某症状，如"肺炎有发热症状"
- TREATED_BY: 疾病由某科室治疗，如"肺炎由呼吸内科治疗"
- USES_TREATMENT: 疾病使用某治疗方法，如"肺炎使用药物治疗"
- REQUIRES_CHECK: 疾病需要某检查，如"肺炎需要胸部CT检查"
- RECOMMENDS_DRUG: 疾病推荐某药物，如"肺炎推荐阿奇霉素"
- COMMONLY_USES_DRUG: 疾病常用某药物，如"肺炎常用青霉素"
- SHOULD_EAT: 疾病宜吃某食物，如"肺炎宜吃鸡蛋"
- SHOULD_NOT_EAT: 疾病不宜吃某食物，如"肺炎不宜吃辣椒"
- RECOMMENDS_RECIPE: 疾病推荐某食谱，如"肺炎推荐百合粥"
- ACCOMPANIES: 疾病伴随其他疾病，如"糖尿病伴随高血压"
- OTHER: 其他关系

仅提取文本中明确提到的实体和关系，不要推断不存在的内容。
除此之外，你还需要遵从一些规则，如：《》、"、""等符号内部的内容皆为一个实体，你不可以将其拆为多个实体。以及你提取实体的目的是为了解决该问题，对于一些不能给解决问题带来帮助的实体你无需输出。
对于无法直接从文本抽取得到关系的情况，为了解决问题，你必须基于已有的实体，推断出实体之间的关系。
你的输出必须是严格的JSON格式，包含两个键："entities"和"relations"。

实体格式必须为：{"name": "实体名称", "type": "实体类型"}
关系格式必须为：{"source": "源实体", "target": "目标实体", "type": "关系类型"}

示例：
输入：肺炎有什么症状？
输出：
{
  "entities": [
    {"name": "肺炎", "type": "Disease"}
  ],
  "relations": [
    {"source": "肺炎", "target": "症状", "type": "HAS_SYMPTOM"}
  ]
}

输入：糖尿病应该吃什么药？
输出：
{
  "entities": [
    {"name": "糖尿病", "type": "Disease"}
  ],
  "relations": [
    {"source": "糖尿病", "target": "药物", "type": "RECOMMENDS_DRUG"}
  ]
}

输入：高血压患者不能吃什么食物？
输出：
{
  "entities": [
    {"name": "高血压", "type": "Disease"}
  ],
  "relations": [
    {"source": "高血压", "target": "食物", "type": "SHOULD_NOT_EAT"}
  ]
}

请严格按照上述格式输出，不要添加任何其他字段。
"""
    
    def _get_answer_generation_prompt(self) -> str:
        """获取答案生成的系统提示词"""
        return """你是一个专业的医学问答助手。你的任务是基于提供的医学知识图谱信息回答用户的医学健康问题。

请遵循以下规则：
1. 仔细阅读用户的问题和提供的医学知识图谱信息
2. 只使用提供的知识图谱信息来回答问题，不要添加知识图谱中没有的信息
3. 如果知识图谱信息不足以回答问题，请明确说明
4. 回答要简洁、准确、专业，使用医学术语但确保通俗易懂
5. 如果问题涉及多个方面（如症状、治疗、饮食等），请分点回答
6. 如果知识图谱信息中有多个相关事实，请整合这些信息
7. 使用中文回答
8. 如果是关于疾病诊断的问题，请提醒用户最终诊断需要咨询专业医生
9. 如果是关于药物使用的问题，请提醒用户需要在医生指导下使用药物

知识图谱信息格式：
- 实体属性：包含疾病、症状、药物、食物等医学实体的各种属性信息
- 关系三元组：包含医学实体之间的关系信息，如疾病-症状、疾病-治疗方法等

请基于这些医学信息，给出准确、专业且负责任的回答。记住要强调任何医学建议都应该在专业医生指导下进行。
"""
     
    def _normalize_entity(self, entity: Dict) -> Dict:
        """统一实体格式"""
        # 如果实体已经是标准格式，直接返回
        if "name" in entity and "type" in entity:
            return entity
             
        # 处理不同格式的实体
        if "text" in entity:
            return {
                "name": entity["text"],
                "type": entity["type"]
            }
        elif "id" in entity and "text" in entity:
            return {
                "name": entity["text"],
                "type": entity["type"]
            }
        else:
            # 如果无法识别格式，返回空实体
            return {"name": "", "type": "Other"}

    def _normalize_relation(self, relation: Dict) -> Dict:
        """统一关系格式"""
        # 如果关系已经是标准格式，直接返回
        if "source" in relation and "target" in relation and "type" in relation:
            return relation
             
        # 处理不同格式的关系
        if "head" in relation and "tail" in relation:
            # 需要从实体映射中获取实际的实体名称
            return {
                "source": relation.get("head", ""),
                "target": relation.get("tail", ""),
                "type": relation["type"]
            }
        else:
            # 如果无法识别格式，返回空关系
            return {"source": "", "target": "", "type": "OTHER"}

    def extract_entities_relations(self, text: str) -> Dict:
        """使用LLM提取实体和关系"""
        self.console.print(Panel(f"[bold blue]问题分析[/bold blue]：\n{text}", 
                                 border_style="blue", expand=False))
        
        # 使用进度指示器
        with self.console.status("[bold green]正在分析问题...", spinner="dots") as status:
            try:
                self.console.print("🔍 正在提取实体和关系...", style="blue")
                
                # 构建完整的提示词
                full_prompt = f"{self.entity_extraction_prompt}\n\n请从以下文本中提取关键实体和实体间的关系:\n\n{text}"
                
                # 调用蓝心大模型
                content = self.call_llm(full_prompt, temperature=0.2)
                

                # 提取JSON部分
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                try:
                    result = json.loads(content)
                    if "entities" not in result or "relations" not in result:
                        raise ValueError("API返回的格式不正确")
                    
                    # 统一实体和关系格式
                    normalized_entities = [self._normalize_entity(e) for e in result["entities"]]
                    normalized_relations = [self._normalize_relation(r) for r in result["relations"]]
                    
                    # 创建实体表格
                    entity_table = Table(title="📊 提取的实体", show_header=True, header_style="bold green")
                    entity_table.add_column("实体名称", style="cyan")
                    entity_table.add_column("实体类型", style="magenta")
                    
                    for entity in normalized_entities:
                        entity_table.add_row(
                            entity.get("name", "未知"),
                            entity.get("type", "未知")
                        )
                    
                    self.console.print(entity_table)
                    
                    # 创建关系表格
                    relation_table = Table(title="🔗 提取的关系", show_header=True, header_style="bold blue")
                    relation_table.add_column("源实体", style="cyan")
                    relation_table.add_column("关系类型", style="yellow")
                    relation_table.add_column("目标实体", style="green")
                    
                    for relation in normalized_relations:
                        relation_table.add_row(
                            relation.get("source", "未知"),
                            relation.get("type", "未知"),
                            relation.get("target", "未知")
                        )
                    
                    self.console.print(relation_table)
                    
                    self.console.print("✅ 实体和关系提取完成!", style="bold green")
                    
                    return {
                        "entities": normalized_entities,
                        "relations": normalized_relations
                    }
                except json.JSONDecodeError:
                    # 如果JSON解析失败，返回空结果
                    self.console.print("❌ JSON解析失败！", style="bold red")
                    return {"entities": [], "relations": []}
                    
            except Exception as e:
                self.console.print(f"❌ 实体关系抽取出错: {str(e)}", style="bold red")
                return {"entities": [], "relations": []}
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示"""
        max_retries = 3
        base_delay = 1.0  # 基础延迟时间（秒）
        
        for attempt in range(max_retries):
            try:
                # 添加延迟以避免频率限制
                if attempt > 0:
                    delay = base_delay * (2 ** attempt)  # 指数退避
                    # 静默等待，不显示等待信息
                    time.sleep(delay)
                
                params = {}
                post_data = {
                    "model_name": "m3e-base",
                    "sentences": [text]
                }
                headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, EMBEDDING_URI, params)
                headers['Content-Type'] = 'application/json'
                
                url = f'https://{DOMAIN}{EMBEDDING_URI}'
                response = requests.post(url, json=post_data, headers=headers)
                
                # 处理429错误（频率限制）
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        # 静默重试，不显示警告信息
                        continue
                    else:
                        raise Exception(f"嵌入API调用频率受限，已达到最大重试次数: {response.status_code}, {response.text}")
                
                if response.status_code != 200:
                    raise Exception(f"嵌入API调用失败: {response.status_code}, {response.text}")
                
                result = response.json()
                
                # 根据实际返回格式解析数据
                if 'data' in result and result['data']:
                    embeddings = result['data']
                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        # 如果data是二维数组，取第一个元素
                        if isinstance(embeddings[0], list):
                            embedding_vector = embeddings[0]
                        else:
                            embedding_vector = embeddings
                    else:
                        raise Exception(f"嵌入API返回数据为空: {result}")
                elif 'embeddings' in result and result['embeddings']:
                    embedding_vector = result['embeddings'][0]
                else:
                    raise Exception(f"嵌入API返回格式错误: {result}")
                

                
                return embedding_vector
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # 静默重试，不显示错误信息
                    continue
                else:
                    self.console.print(f"❌ 获取向量表示出错: {str(e)}", style="bold red")
                    return []
        
        # 如果所有重试都失败，返回空列表
        return []
    
    def call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """调用蓝心大模型，带重试机制"""
        max_retries = 3
        base_delay = 1.0  # 基础延迟时间（秒）
        
        for attempt in range(max_retries):
            try:
                # 添加延迟以避免频率限制
                if attempt > 0:
                    delay = base_delay * (2 ** attempt)  # 指数退避
                    # 静默等待，不显示等待信息
                    time.sleep(delay)
                
                # 调用蓝心大模型
                params = {
                    'requestId': str(uuid.uuid4())
                }
                
                data = {
                    'prompt': prompt,
                    'model': 'vivo-BlueLM-TB-Pro',
                    'sessionId': str(uuid.uuid4()),
                    'extra': {
                        'temperature': temperature
                    }
                }
                
                headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, LLM_URI, params)
                headers['Content-Type'] = 'application/json'
                
                url = f'https://{DOMAIN}{LLM_URI}'
                response = requests.post(url, json=data, headers=headers, params=params)
                
                # 处理429错误（频率限制）
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        # 静默重试，不显示警告信息
                        continue
                    else:
                        raise Exception(f"LLM API调用频率受限，已达到最大重试次数: {response.status_code}, {response.text}")
                
                if response.status_code != 200:
                    raise Exception(f"LLM API调用失败: {response.status_code}, {response.text}")
                    
                res_obj = response.json()
                if res_obj['code'] != 0 or not res_obj.get('data'):
                    raise Exception(f"LLM API返回错误: {res_obj}")
                
                content = res_obj['data']['content']
                
                return content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # 静默重试，不显示错误信息
                    continue
                else:
                    raise e
        
        # 这行实际上不会被执行，因为最后一次重试会抛出异常
        raise Exception("LLM调用失败")
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def query_neo4j(self, entities: List[Dict], relations: List[Dict]) -> Dict:
        """查询Neo4j数据库"""
        self.console.print(Panel("[bold green]知识图谱查询[/bold green]", border_style="green", expand=False))
        
        result = {
            "entity_properties": [],
            "related_triples": []
        }
        
        # 存储已查询过的实体，避免重复查询
        processed_entities = set()
        
        # 存储需要进行多跳查询的实体及其相似度
        entities_for_multi_hop = []
        
        # 获取问题的向量表示，用于计算实体相似度
        question_embedding = self.get_embedding(" ".join([e.get("name", "") for e in entities] + [r.get("type", "") for r in relations]))
        
        with self.console.status("[bold blue]正在查询知识图谱...", spinner="dots") as status:
            # 1. 查询实体属性
            self.console.print("🔍 正在查询实体属性...", style="blue")
            for entity in entities:
                # 检查实体字典中是否包含必要的键
                if "name" not in entity:
                    self.console.print(f"⚠️ 警告：实体缺少name属性: {entity}", style="yellow")
                    continue
                    
                entity_name = entity.get("name")
                entity_type = entity.get("type", "Other")
                
                # 添加到已处理实体集合
                processed_entities.add(entity_name)
                
                self.console.print(f"  查询实体: [cyan]{entity_name}[/cyan] ([magenta]{entity_type}[/magenta])")
                
                # 根据节点类型构建查询
                if entity_type == "Disease":
                    query = f"""
                    MATCH (n:Disease {{name: $name}})
                    RETURN n LIMIT {self.search_budget['entity_limit']}
                    """
                else:
                    query = f"""
                    MATCH (n {{name: $name}})
                    WHERE any(label in labels(n) WHERE label in ['Disease', 'Category', 'Symptom', 'Department', 'Treatment', 'Check', 'Drug', 'Food', 'Recipe'])
                    RETURN n LIMIT {self.search_budget['entity_limit']}
                    """
                
                try:
                    nodes = self.graph.run(query, name=entity_name).data()
                    if nodes:
                        self.console.print(f"  ✅ 找到 [bold]{len(nodes)}[/bold] 个匹配实体")
                        for node in nodes:
                            # 获取节点的所有属性
                            properties = dict(node["n"])
                            
                            # 添加到结果
                            result["entity_properties"].append({
                                "name": entity_name,
                                "type": entity_type,
                                "properties": properties
                            })
                            
                            # 计算实体与问题的相似度
                            entity_embedding = self.get_embedding(entity_name)
                            similarity = self.calculate_similarity(question_embedding, entity_embedding)
                            
                            # 添加到多跳查询候选列表
                            entities_for_multi_hop.append({
                                "name": entity_name,
                                "similarity": similarity
                            })
                    else:
                        self.console.print(f"  ⚠️ 未找到实体: [cyan]{entity_name}[/cyan]", style="yellow")
                except Exception as e:
                    self.console.print(f"  ❌ 查询实体属性出错: {str(e)}", style="bold red")
            
            # 2. 查询关系三元组
            self.console.print("🔍 正在查询关系三元组...", style="blue")
            for relation in relations:
                # 检查关系字典中是否包含必要的键
                if not all(key in relation for key in ["source", "target", "type"]):
                    self.console.print(f"⚠️ 警告：关系缺少必要属性: {relation}", style="yellow")
                    continue
                    
                source = relation["source"]
                target = relation["target"]
                rel_type = relation["type"]
                
                self.console.print(f"  查询关系: [cyan]{source}[/cyan] --[yellow]{rel_type}[/yellow]--> [green]{target}[/green]")
                
                # 获取关系类型的向量表示
                rel_embedding = self.get_embedding(rel_type)
                
                # 查询所有可能的关系三元组
                query = f"""
                MATCH (s)-[r]->(t)
                WHERE s.name = $source
                RETURN s, r, t LIMIT {self.search_budget['relation_limit']}
                UNION
                MATCH (s)-[r]->(t)
                WHERE t.name = $target
                RETURN s, r, t LIMIT {self.search_budget['relation_limit']}
                """
                
                try:
                    triples = self.graph.run(query, source=source, target=target).data()
                    if triples:
                        self.console.print(f"  ✅ 找到 [bold]{len(triples)}[/bold] 个匹配三元组")
                        # 计算相似度并排序
                        scored_triples = []
                        for triple_data in triples: # Renamed to avoid conflict with outer 'triple'
                            # 获取关系的向量表示
                            current_rel_type = type(triple_data["r"]).__name__
                            current_entity_name = triple_data["s"].get("name") or triple_data["t"].get("name", "")
                            current_embedding = self.get_embedding(f"{current_entity_name} {current_rel_type}")
                            
                            # 计算相似度
                            similarity = self.calculate_similarity(question_embedding, current_embedding)
                            
                            scored_triples.append({
                                "similarity": similarity,
                                "source": dict(triple_data["s"]),
                                "relation": current_rel_type,
                                "target": dict(triple_data["t"])
                            })
                        
                        # 按相似度排序
                        scored_triples.sort(key=lambda x: x["similarity"], reverse=True)
                        
                        # 添加相似度最高的三元组
                        if scored_triples:
                            # 添加相似度最高的前k个三元组
                            top_triples = scored_triples[:self.search_budget['top_k_triples']]
                            for idx, top_triple_item in enumerate(top_triples): # Renamed to avoid conflict
                                result["related_triples"].append(top_triple_item)
                                source_name = top_triple_item['source'].get('name', '')
                                target_name = top_triple_item['target'].get('name', '')  
                                
                                self.console.print(  
                                    f"  👍 匹配 #{idx+1}: [cyan]{source_name}[/cyan] --[yellow]{top_triple_item['relation']}[/yellow]--> [green]{target_name}[/green] (相似度: {top_triple_item['similarity']:.2f})"  
                                )
                    else:
                        self.console.print(f"  ⚠️ 未找到关系三元组", style="yellow")
                
                except Exception as e:
                    self.console.print(f"  ❌ 查询关系三元组出错: {str(e)}", style="bold red")
            
            # 3. 查询与实体相连的其他实体（第一跳）
            self.console.print("🔍 正在查询相连实体（第一跳）...", style="blue")
            for entity in entities:
                # 检查实体字典中是否包含必要的键
                if "name" not in entity:
                    continue
                    
                entity_name = entity.get("name")
                
                self.console.print(f"  查询与 [cyan]{entity_name}[/cyan] 相连的实体")
                
                # 查询与该实体相连的所有其他实体
                query1= f"""
                MATCH (n)-[r]->(m)  
                WHERE n.name = $name  
                AND any(label IN labels(m) WHERE label IN ['Disease', 'Category', 'Symptom', 'Department', 'Treatment', 'Check', 'Drug', 'Food', 'Recipe'])  
                RETURN n, r, m, type(r) AS rel_type  
                LIMIT {self.search_budget['one_hop_limit']}  
                """
                query2= f""" 
                MATCH (n)<-[r]-(m)  
                WHERE n.name = $name  
                AND any(label IN labels(m) WHERE label IN ['Disease', 'Category', 'Symptom', 'Department', 'Treatment', 'Check', 'Drug', 'Food', 'Recipe'])  
                RETURN n, r, m, type(r) AS rel_type  
                LIMIT {self.search_budget['one_hop_limit']}
                """
                
                try:
                    results1 = self.graph.run(query1, name=entity_name).data()  
                    grouped = defaultdict(list)  
                    for record in results1:  
                        rel_type = record['rel_type']  
                        if len(grouped[rel_type]) < 5:  
                            grouped[rel_type].append(record)  
                    # 再把所有分组的结果合并为最终结果  
                    final_results1 = []  
                    for rel_list in grouped.values():  
                        final_results1.extend(rel_list) 
                    connected_triples1 = final_results1
                    results2 = self.graph.run(query2, name=entity_name).data()  
                    grouped = defaultdict(list)  
                    for record in results2:  
                        rel_type = record['rel_type']  
                        if len(grouped[rel_type]) < 5:  
                            grouped[rel_type].append(record)  
                    # 再把所有分组的结果合并为最终结果  
                    final_results2 = []  
                    for rel_list in grouped.values():  
                        final_results2.extend(rel_list) 
                    connected_triples2 = final_results2
                    
                    if connected_triples1 or connected_triples2:
                        self.console.print(f"  ✅ 找到 [bold]{len(connected_triples1)}[/bold] 个相连实体")
                        self.console.print(f"  ✅ 找到 [bold]{len(connected_triples2)}[/bold] 个被相连实体")
                        for triple in connected_triples1:
                            # 获取关系的向量表示
                            rel_type = type(triple["r"]).__name__
                            
                            # 计算简单相似度，不进行向量计算
                            similarity = 0.5  # 默认相似度
                            
                            result["related_triples"].append({
                                "similarity": similarity,
                                "source": dict(triple["n"]),
                                "relation": rel_type,
                                "target": dict(triple["m"])
                            })
                            
                            # 获取实体名称
                            source_name = triple["n"].get("name", "未知")
                            target_name = triple["m"].get("name", "未知")
                            
                            # 如果目标实体未处理过，计算其与问题的相似度
                            if target_name not in processed_entities:
                                processed_entities.add(target_name)
                                target_embedding = self.get_embedding(target_name)
                                target_similarity = self.calculate_similarity(question_embedding, target_embedding)
                                entities_for_multi_hop.append({
                                    "name": target_name,
                                    "similarity": target_similarity
                                })
                            
                            self.console.print(f"  👉 相连实体: [cyan]{source_name}[/cyan] --[yellow]{rel_type}[/yellow]--> [green]{target_name}[/green]")
                        for triple in connected_triples2:
                            # 获取关系的向量表示
                            rel_type = type(triple["r"]).__name__
                            
                            # 计算简单相似度，不进行向量计算
                            similarity = 0.5  # 默认相似度
                            
                            result["related_triples"].append({
                                "similarity": similarity,
                                "source": dict(triple["m"]),
                                "relation": rel_type,
                                "target": dict(triple["n"])
                            })
                            
                            # 获取实体名称
                            source_name = triple["m"].get("name", "未知")
                            target_name = triple["n"].get("name", "未知")
                            
                            # 如果目标实体未处理过，计算其与问题的相似度
                            if target_name not in processed_entities:
                                processed_entities.add(target_name)
                                target_embedding = self.get_embedding(target_name)
                                target_similarity = self.calculate_similarity(question_embedding, target_embedding)
                                entities_for_multi_hop.append({
                                    "name": target_name,
                                    "similarity": target_similarity
                                })
                            
                            self.console.print(f"  👉 相连实体: [cyan]{source_name}[/cyan] --[yellow]{rel_type}[/yellow]--> [green]{target_name}[/green]")
                    else:
                        self.console.print(f"  ⚠️ 未找到相连实体", style="yellow")
                
                except Exception as e:
                    self.console.print(f"  ❌ 查询相连实体出错: {str(e)}", style="bold red")
            
            # 4. 多跳查询 - 选择相似度最高的前10个实体进行第二跳查询
            if self.enable_multi_hop and entities_for_multi_hop:
                # 按相似度排序并选择前10个
                entities_for_multi_hop.sort(key=lambda x: x["similarity"], reverse=True)
                top_entities = entities_for_multi_hop[:self.search_budget['top_k_multi_hop_entities']]
                
                self.console.print(Panel("[bold yellow]多跳查询（第二跳）[/bold yellow]", border_style="yellow", expand=False))
                self.console.print("🔍 选择以下实体进行第二跳查询:", style="blue")
                for idx, entity in enumerate(top_entities):
                    self.console.print(f"  {idx+1}. [cyan]{entity['name']}[/cyan] (相似度: {entity['similarity']:.2f})")
                
                # 对每个高相似度实体进行第二跳查询
                for entity in top_entities:
                    entity_name = entity["name"]
                    
                    self.console.print(f"  查询与 [cyan]{entity_name}[/cyan] 相连的实体（第二跳）")
                    
                    # 查询与该实体相连的所有其他实体
                    query = f"""
                    MATCH (n)-[r]->(m)
                    WHERE n.name = $name
                    AND any(label in labels(m) WHERE label in ['Disease', 'Category', 'Symptom', 'Department', 'Treatment', 'Check', 'Drug', 'Food', 'Recipe'])
                    RETURN n, r, m LIMIT {self.search_budget['multi_hop_limit']}
                    UNION  
                    MATCH (n)<-[r]-(m)  
                    WHERE n.name = $name  
                    AND any(label IN labels(m) WHERE label IN ['Disease', 'Category', 'Symptom', 'Department', 'Treatment', 'Check', 'Drug', 'Food', 'Recipe'])  
                    RETURN n, r, m  
                    LIMIT {self.search_budget['multi_hop_limit']}
                    """
                    
                    try:
                        connected_triples = self.graph.run(query, name=entity_name).data()
                        if connected_triples:
                            self.console.print(f"  ✅ 找到 [bold]{len(connected_triples)}[/bold] 个相连实体（第二跳）")
                            for triple in connected_triples:
                                # 获取关系的向量表示
                                rel_type = type(triple["r"]).__name__
                                
                                # 获取实体名称
                                source_name = triple["n"].get("name", "未知")
                                target_name = triple["m"].get("name", "未知")
                                
                                # 如果目标实体未处理过，则添加到结果
                                if target_name not in processed_entities:
                                    processed_entities.add(target_name)
                                    
                                    # 计算与问题的相似度
                                    target_embedding = self.get_embedding(target_name)
                                    target_similarity = self.calculate_similarity(question_embedding, target_embedding)
                                    
                                    # 添加到结果
                                    result["related_triples"].append({
                                        "similarity": target_similarity,
                                        "source": dict(triple["n"]),
                                        "relation": rel_type,
                                        "target": dict(triple["m"]),
                                        "hop": 2  # 标记为第二跳查询结果
                                    })
                                    
                                    self.console.print(f"  👉 第二跳实体: [cyan]{source_name}[/cyan] --[yellow]{rel_type}[/yellow]--> [green]{target_name}[/green] (相似度: {target_similarity:.2f})")
                        else:
                            self.console.print(f"  ⚠️ 未找到第二跳相连实体", style="yellow")
                    
                    except Exception as e:
                        self.console.print(f"  ❌ 查询第二跳实体出错: {str(e)}", style="bold red")
            
            # 5. 按相似度排序所有关系三元组
            result["related_triples"].sort(key=lambda x: x["similarity"], reverse=True)
            
            # 显示查询结果摘要
            self.console.print("✅ 知识图谱查询完成!", style="bold green")
            self.console.print(f"📊 查询结果: {len(result['entity_properties'])} 个实体, {len(result['related_triples'])} 个关系三元组", style="bold blue")
            
            # 统计多跳查询的结果
            if self.enable_multi_hop:
                second_hop_count = sum(1 for triple in result["related_triples"] if triple.get("hop") == 2)
                if second_hop_count > 0:
                    self.console.print(f"🔄 其中包含 {second_hop_count} 个第二跳查询结果", style="bold yellow")
        
        return result
    
    def generate_answer(self, question: str, knowledge: Dict) -> str:
        """生成答案"""
        self.console.print(Panel("[bold purple]生成回答[/bold purple]", border_style="purple", expand=False))
        
        with self.console.status("[bold green]正在生成回答...", spinner="dots") as status:
            try:
                # 限制知识图谱信息的数量以避免提示词过长
                max_entities = 10  # 最多10个实体
                max_triples = 20   # 最多20个关系三元组
                
                # 截取实体属性信息
                limited_entities = knowledge['entity_properties'][:max_entities]
                
                # 截取关系三元组信息（按相似度排序，取前20个）
                limited_triples = knowledge['related_triples'][:max_triples]
                
                # 简化实体和关系信息的表示
                entities_summary = []
                for entity in limited_entities:
                    # 只保留关键属性，简化信息
                    simplified_entity = {
                        "name": entity.get("name", ""),
                        "type": entity.get("type", ""),
                        "key_properties": {k: v for k, v in entity.get("properties", {}).items() 
                                         if k in ["name", "description", "category", "type"] and len(str(v)) < 100}
                    }
                    entities_summary.append(simplified_entity)
                
                triples_summary = []
                for triple in limited_triples:
                    # 简化关系三元组表示
                    simplified_triple = {
                        "source": triple.get("source", {}).get("name", ""),
                        "relation": triple.get("relation", ""),
                        "target": triple.get("target", {}).get("name", ""),
                        "similarity": round(triple.get("similarity", 0.0), 2)
                    }
                    triples_summary.append(simplified_triple)
                
                # 构建简化的提示词
                full_prompt = f"""{self.answer_generation_prompt}

问题：{question}

知识图谱信息：
相关实体（共{len(entities_summary)}个）：
{json.dumps(entities_summary, ensure_ascii=False, indent=2)}

相关关系（共{len(triples_summary)}个，按相似度排序）：
{json.dumps(triples_summary, ensure_ascii=False, indent=2)}

请基于以上医学知识图谱信息回答问题。如果信息不足以回答问题，请说明。"""

                # 检查提示词长度
                prompt_length = len(full_prompt)
                self.console.print(f"📏 提示词长度: {prompt_length:,} 字符", style="blue")
                
                # 如果提示词仍然太长，进一步缩减
                if prompt_length > 8000:  # 设置一个安全阈值
                    self.console.print("⚠️ 提示词过长，进一步缩减信息...", style="yellow")
                    
                    # 进一步减少数量
                    max_entities = 5
                    max_triples = 10
                    
                    limited_entities = knowledge['entity_properties'][:max_entities]
                    limited_triples = knowledge['related_triples'][:max_triples]
                    
                    # 重新构建更简化的提示词
                    entities_text = "; ".join([f"{e.get('name', '')}({e.get('type', '')})" for e in limited_entities])
                    triples_text = "; ".join([f"{t.get('source', {}).get('name', '')}-{t.get('relation', '')}-{t.get('target', {}).get('name', '')}" for t in limited_triples])
                    
                    full_prompt = f"""{self.answer_generation_prompt}

问题：{question}

知识图谱信息：
相关实体：{entities_text}
相关关系：{triples_text}

请基于以上医学知识图谱信息回答问题。"""
                    
                    self.console.print(f"📏 缩减后提示词长度: {len(full_prompt):,} 字符", style="blue")
                
                self.console.print("🧠 蓝心大模型思考中...", style="blue")
                
                # 调用蓝心大模型
                answer = self.call_llm(full_prompt)
                

                self.console.print("✅ 回答生成完成!", style="bold green")
                
                return answer
                
            except Exception as e:
                self.console.print(f"❌ 生成答案出错: {str(e)}", style="bold red")
                return "抱歉，我无法回答这个问题。"
    
    def answer_question(self, question: str) -> str:
        """回答问题的主函数"""
        # 1. 提取实体和关系
        self.console.print(Panel(f"[bold]💡 问题[/bold]: {question}", 
                                 title="医学知识图谱问答系统",
                                 border_style="cyan", 
                                 expand=False))
        
        extraction_result = self.extract_entities_relations(question)
        
        # 2. 查询Neo4j数据库
        knowledge = self.query_neo4j(
            extraction_result["entities"],
            extraction_result["relations"]
        )
        
        # 3. 生成答案
        answer = self.generate_answer(question, knowledge)
        
        # 4. 展示答案
        self.console.print(Panel(Markdown(answer), 
                                title="📝 回答", 
                                border_style="green", 
                                expand=False))
        
        return answer

def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="基于知识图谱和LLM的问答系统")
    parser.add_argument("--neo4j_uri", type=str, default=os.getenv("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j数据库URI")
    parser.add_argument("--neo4j_user", type=str, default=os.getenv("NEO4J_USER", "neo4j"), help="Neo4j用户名")
    parser.add_argument("--neo4j_password", type=str, default=os.getenv("NEO4J_PASSWORD", "123456789"), help="Neo4j密码")
    parser.add_argument("--disable_multi_hop", action="store_false", dest="enable_multi_hop", help="禁用多跳查询功能 (默认为启用)")
    parser.add_argument("--search_budget", type=str, default="Deeper", choices=["Deeper", "Deep"], help="设置搜索预算模式 (Deeper, Deep)，默认为 Deeper")
    parser.set_defaults(enable_multi_hop=True)
    args = parser.parse_args()

    # 打印欢迎信息
    console.print(Panel(
        "[bold cyan]医学知识图谱问答系统[/bold cyan]\n\n"
        "基于Neo4j医学知识图谱和蓝心大模型的医学健康问答系统\n"
        "[yellow]⚠️ 本系统仅供参考，任何医学建议都应在专业医生指导下进行[/yellow]",
        border_style="cyan",
        title="欢迎",
        subtitle="v1.0"
    ))
    
    # 蓝心大模型使用内置的APP_ID和APP_KEY，无需额外配置
    
    # 初始化RAG系统
    rag_system = Neo4jRAGSystem(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        enable_multi_hop=args.enable_multi_hop,
        search_budget_mode=args.search_budget
    )
    
    console.print(f"🚀 基于医学知识图谱的问答系统已启动。多跳查询已 {'[bold green]启用[/bold green]' if args.enable_multi_hop else '[bold red]禁用[/bold red]'}。搜索预算: [bold magenta]{args.search_budget}[/bold magenta]。输入'退出'结束对话。", style="bold green")
    
    # 交互式问答
    while True:
        question = input("\n请输入问题：")
        if question.lower() in ['退出', 'exit', 'quit']:
            console.print("👋 感谢使用！再见！", style="bold cyan")
            break
        
        answer = rag_system.answer_question(question)

if __name__ == "__main__":
    main()