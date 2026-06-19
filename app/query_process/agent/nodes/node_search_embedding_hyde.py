import time
import sys

from langchain_core.messages import HumanMessage

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import  add_done_task,add_running_task
from app.core.logger import logger


def step2_create_hyde_answer(rewritten_query):
    """
    使用大模型生成假设答案
    :param rewritten_query: 重写后的问题
    :return: 生成的假设性答案
    """
    #调用提示词
    prompt=load_prompt("hyde_prompt",rewritten_query=rewritten_query)
    messages=[
        HumanMessage(content=prompt)
    ]
    #创建大模型
    llm=get_llm_client()
    answer=llm.invoke( messages)
    result=answer.content
    logger.info(f"HyDE 模型生成答案：{result}")
    return result

def step3_search_embedding(hyde_answer, rewritten_query,item_names):
    """
    根据大模型生成的假设答案和问题进行拼接进行向量化操作
    :param hyde_answer:
    :param rewritten_query:
    :return:
    """
    #1.拼接问题和答案
    query_str = rewritten_query + hyde_answer
    #2.对拼接完的进行向量化操作
    embeddings=generate_embeddings([query_str])
    #3.生成查询AnnSearchRequest列表
    item_name_str = ', '.join(f'"{item}"' for item in item_names)
    final_result=create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=f"item_name in [{item_name_str}]"
    )
    #4.进行查询比对
    milvus_client = get_milvus_client()
    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=final_result,
        ranker_weights=(0.9, 0.1),
        output_fields=["item_name", "content", "title", "parent_title", "chunk_id"]
    )
    # 5.处理返回结果
    result = resp[0] if resp else []
    logger.info(f"假设性问题检索结果：{result}")
    return result

def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    logger.info("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设性答案
    #1.提取参数
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    #2.使用LLM生成假设性答案
    hyde_answer = step2_create_hyde_answer(rewritten_query)
    #3.将假设性答案和问题拼接完事转成向量
    embeddings_query = step3_search_embedding(hyde_answer, rewritten_query,item_names)
    #4. 赋值然后返回结果
    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info("---HyDE 处理结束---")
    return {"hyde_embedding_chunks":[]}





if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "华为B3-211H显示器怎么操作？",
        "rewritten_query": "华为B3-211H显示器的具体操作步骤是什么？",
        "item_names": ["华为B3-211H显示器"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print(result)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")