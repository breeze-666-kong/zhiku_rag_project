import sys
from app.utils.task_utils import *

from dotenv import load_dotenv
import sys
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
from app.utils.task_utils import add_running_task

load_dotenv()
# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5 # 最大间断分值


def step_1_merge_rrf_mcp(state):
    """
    对两个非同源的结果进行合并 rrf+mcp数据整合
    :param state:
    :return:
    """
    #1.先拿出两个数据
    rrf_chunks= state.get("rrf_chunks", [])
    web_search_docs= state.get("web_search_docs", [])
    #结果容器
    result = []
    #2.通过遍历把rrf传进集合
    for chunk in rrf_chunks:
        entity=chunk.get("entity")
        chunk_id=chunk.get("chunk_id")
        content=entity.get("content")
        title=entity.get("title")
        result.append({
            "chunk_id": chunk_id,
            "text": content,
            "title": title,
            "source":"local",
            "url":""
        })
    #3.通过遍历把mcp传进集合
    for doc in web_search_docs:
        text=doc.get("snippet")
        title=doc.get("title")
        url=doc.get("url")
        result.append({
            "chunk_id": None,
            "text": text,
            "title": title,
            "source":"web",
            "url":url
        })
    logger.info(f"多路数据融合，最终结果为:{result}")
    return result

def step_2_rerank_doc_list(doc_list, state):
    """
    使用rerank进行精排
    :param doc_list:
    :param state:
    :return:
    """
    #1.获得原有问题
    rewritten_query=state.get("rewritten_query")or state.get("original_query")
    #2.获取问题对应答案
    text_list=[doc.get("text") for doc in doc_list]
    #3.加载rerank模型
    rerank_model = get_reranker_model()
    #4.处理数据 设置 问题 + 答案 成对 -》 装到列表中，调用打分方法
    # [   [问题,答案]  , (问题，答案) -》 512]
    question_pairs=[[rewritten_query, text] for text in text_list]
    scores=rerank_model.compute_score(question_pairs,normalize= True)#normalize是将分缩放到0-1之间,防止出现分差特别大的情况
    #5.将原有数据添加分数  就是第一步中的result里加分数属性
    doc_list_with_scores=[]
    for score,item in zip(scores,doc_list):
        item["score"]=score
        doc_list_with_scores.append(item)
    # 排序
    doc_list_with_scores.sort(key=lambda x:x['score'],reverse=True)
    logger.info(f"已经完成排序和打分！最终结果为：{doc_list_with_scores}")
    return doc_list_with_scores

def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    对不同源的结果合并并使用rerank模型打分
    节点作用： rrf + mcp -> 精排序 rerank -> chunk - 打分  -> 算法 -> top k
    """
    logger.info("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    #1.对不同原结果合并到一个集合中
    """
          [
             rrf = {id:chunk_id,distance:0.x,entity:{chunk_id,content,title}}
             mcp = {snippet: 内容,title:标题,url:关联的文章或者图片的地址}
             {
                text:内容 snippet content,
                chunk_id: chunk_id rrf有  mcp None,
                title: title ,
                url : rrfNone mcp url ,
                source: web -> mcp  || local -> rrf 
             }
          ]
        """
    doc_list = step_1_merge_rrf_mcp(state)
    # 2. 启用rerank进行精排 （数据和分）
    """
    [
      {
            text:内容 snippet content,
            chunk_id: chunk_id rrf有  mcp None,
            title: title ,
            url : rrfNone mcp url ,
            source: web -> mcp  || local -> rrf ,
            score: rerank打的分 
      }
    ]
    """
    rerank_score_list = step_2_rerank_doc_list(doc_list, state)
    # 3. 启动算法进行放断崖以及topk处理

    # 4. 结果装到state中即可



    logger.info("---Rerank处理结束---")
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"entity":{"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9}},
        {"entity":{"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8}},
        {"entity":{"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}}  # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")