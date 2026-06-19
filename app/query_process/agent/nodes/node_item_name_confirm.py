import json
import time
import sys

from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from langchain_core.messages import HumanMessage

from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.core.load_prompt import load_prompt
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task



def step_3_llm_item_name_and_rewrite_query(original_query, history_messages):
    """
    根据历史记录来重写问题和识别item_name
    :param original_query:原问题
    :param history_messages:历史消息
    :return:{  item_name = [] , rewritten_query:问题
              }
    """
    #1.准备提示词
    history_text=""
    for history_message in history_messages:
        history_text += (f"聊天角色：{history_message['role']}，"
                         f"回答内容： {history_message['text']}，"
                         f"重写问题： {history_message['rewritten_query']}，"
                         f"关联主体： {','.join(history_message.get('item_names', []))},"
                         f"时间： {history_message['ts']}\n")
    prompt= load_prompt("rewritten_query_and_itemnames" ,history_text=history_text,
                        query=original_query )
    #2,掉大模型
    lm_client=get_llm_client(json_mode=True)
    messages= [
        HumanMessage(content=prompt)
    ]
    response = lm_client.invoke(messages)
    #3.解析结果
    content= response.content
    #防止大模型写的时候有前缀```json    ```
    if content.startswith("```json"):
        content = content.replace("```json","").replace("```","")
    dict_content = json.loads(content)
    if "item_names" not in dict_content:
        dict_content["item_names"] = []
    if "rewritten_query" not in dict_content:
        dict_content["rewritten_query"] = original_query
    #4.返回数据
    logger.info(f"已经完成问题的重写和item_name的提取！ 结果为：{dict_content}")
    return dict_content


def step_4_query_milvus_item_names(item_names):
    """
    查询向量数据库!进行item_name的确定
    :param item_names:模型提取的item_name可能不准！
    :return:
     [{extracted:模型item_name,matches:[{item_name:xx,score:0.9...}]}]
    """
    final_result=[]
    #1.获得向量数据库客户端
    milvus_client = get_milvus_client()
    #2.将item_name转成稠密和稀疏变量
    embeddings= generate_embeddings(item_names)
    #3.混合查询
    for index, item_name in enumerate(item_names):
        #获得对应向量
        dense_vector= embeddings["dense"][index]
        sparse_vector = embeddings["sparse"][index]
        #拼对应的AnnSearchRequest
        reqs=create_hybrid_search_requests(
            dense_vector = dense_vector,
            sparse_vector = sparse_vector )
        #定义混合检索和定义权重
        response= hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.8,0.2),
            norm_score=True
        )
        """
         [
            [
              {id:xx , distance: 0.x,entity:{item_name:xxx} } ,
              {id:xx , distance: 0.x,entity:{item_name:xxx} } 
             ]
          ]
        """

        #解析结果
        matches=[]
        if response and len(response) > 0:
            for item in response[0]:
                score= item.get("distance", 0)
                entity= item.get("entity", {})
                new_item_name= entity.get("item_name", "")
                if new_item_name:
                    matches.append({"item_name": new_item_name, "score": score})
    #4.提取查询结果
        final_result.append({"extracted": item_name, "matches": matches})
    #5.返回数据
    logger.info(f"查询向量数据库结果为：{final_result}")
    return final_result


def step_5_item_name_confirm(query_milvus_results):
    """
    通过向量数据库查询item_name 根据分数来算出确定和可选的item_name
    大于0.85的确认,大于0.6的可选
    :param query_milvus_results:
    :return:
    """
    #1.准备两个列表 确认和可选的
    confirmed_item_names = []  # 确定
    options_item_names = []  # 可选
    #2.循环处理元素query_milvus_results
    for item_name_meta in query_milvus_results:
        extracted_name = item_name_meta.get("extracted")
        matches = item_name_meta.get("matches", [])
        #3.对分数排序使用倒叙
        matches.sort(key=lambda x: x.get("score",0.0), reverse=True)
        #对高分处理
        high_score_matches = [x for x in matches if x.get("score", 0.0)>= 0.85]
        #中等分处理  在这里暂时不考虑大于0.6也大于0.85的分,因为对高分处理的话就不会对中等分处理,在后续逻辑有体现
        middle_score_matches = [x for x in matches if x.get("score", 0) >= 0.6]
        #4.高分处理结果  只有一个高分和多个高分分别处理
        if len(high_score_matches) == 1:
            confirmed_item_names.append(high_score_matches[0].get("item_name"))
            continue
        if len(high_score_matches) > 1:
            #优先考虑名字一样的
            same_name_item= None
            for item in high_score_matches:
                if item.get("item_name") == extracted_name:
                    same_name_item= item
                    break
            if not same_name_item:
                # 没有名字一样的,则使用分数最高的
                same_name_item = high_score_matches[0]
            confirmed_item_names.append(high_score_matches[0].get("item_name"))
            continue
        #5.中分处理
        if len(middle_score_matches)>0:
            for item in middle_score_matches[:2]:
                options_item_names.append(item.get("item_name"))
            continue
        logger.info(f"没有匹配的item_name，忽略：{extracted_name}")
        #6.返回结果 去重复操作
    result = {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options_item_names": list(set(options_item_names))
    }
    logger.info(f"处理结果为：{result}")
    return result


def step_6_deal_list(state, item_results, history, rewritten_query):
    """
    根据集合的数据来判定是否对answer进行赋值
    :param state:
    :param item_results:# result = {
        #         "confirmed_item_names":list(set(confirmed_item_names)),
        #         "options_item_names":list(set(options_item_names))
    :param history:历史聊天对话
    :param rewritten_query:改写完的问题
    :return:
    """
    #1.获取两个集合  确认的和可选的
    confirmed_item_names = item_results.get("confirmed_item_names", [])
    options_item_names = item_results.get("options_item_names", [])
    #2.确认集合处理
    if len(confirmed_item_names)>0:
        state["item_names"] = confirmed_item_names
        state["rewritten_query"] = rewritten_query
        state["history"] = history
        if "answer" in state:
            del state["answer"]
        logger.info(f"有确定的item_name:{confirmed_item_names}")
        return state
    #3.确认可选集合
    if len(options_item_names)>0:
        options_name=",".join(options_item_names)
        answer= f"请您选择一个商品名称，可选名称有：{options_name}"
        state["answer"]=  answer
        logger.info(f"有可选的item_name:{options_item_names}")
        return state
    #4.都没有
    answer= "请您重新输入问题，我无法理解"
    state["answer"]= answer
    logger.info(f"没有匹配的item_name，忽略：{state['original_query']}")
    return state

def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    主要目的:重写用户提出的问题为了解决,语义歧义,补全上下文,去除口头话语言,润色一下让其提高召回率
    1.获取历史消息
    2.保存本次聊天消息
    3.使用大模型提取item_name并且重写问题
    4.使用向量数据库来查询
    5.对item_name结果进行打分分类处理 A 【确认集合】  B【可选集合】
    6.对比分数 高分->下个节点  中分->看看备选,询问是不是  低分->说不知道啥意思
    7.补充state状态 item_names rewritten_query  history
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    logger.info("开始执行节点：node_item_name_confirm")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    #1.获取历史消息
    history = get_recent_messages(state["session_id"])
    # # 2.保存本次聊天消息,放到最后了,因为在这里的item_names是没有任何东西的
    # message= save_chat_message(
    #     session_id=state["session_id"],
    #     role="user",
    #     text=state["original_query"],
    #     rewritten_query=state.get("rewritten_query", ""),
    #     item_names=state.get("item_names", []),
    #     image_urls=state.get("image_urls", [])
    # )
    #3.使用大模型提取item_name并且重写问题
    item_names_and_rewritten_query= step_3_llm_item_name_and_rewrite_query(state["original_query"],history)
    item_names= item_names_and_rewritten_query.get("item_names", [])
    rewritten_query= item_names_and_rewritten_query.get("rewritten_query", "")
    item_results = {}
    #4.使用向量数据库来查询
    if len(item_names) > 0:
        #有提取出来的item_name
        #返回： 1 -> 向量数据库中item_names (向量查询) 2 -> 向量数据库中item_names (向量查询)
        #  [ { extracted:（模型提取的item_name）, matches:[{item_name:名字,score:0.8},{item_name:名字,score:0.8}]  }，
        #  { extracted:（模型提取的item_name）, matches:[{item_name:名字,score:0.8},{item_name:名字,score:0.8}]  }，
        #      ]
        query_milvus_results = step_4_query_milvus_item_names(item_names)
        #5.查询结果进行处理 区分 确定的item_name 以及可选的item_name  -》  没有对应的item_name
        # result = {
        #         "confirmed_item_names":list(set(confirmed_item_names)),
        #         "options_item_names":list(set(options_item_names))
        #     }
        item_results = step_5_item_name_confirm(query_milvus_results)
    #6.根据item_name确定的集合来对问题进行返回处理->answer的赋值
    # 参数： item_results （两个集合） || 修改历史聊天记录对应item_names history_chats
    state = step_6_deal_list(state, item_results, history, rewritten_query)
    # 7. 记录本次的聊天对话 （answer回答）
    save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
        image_urls=[]
    )
    # 后面会调用大模型，进行逻辑处理
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    logger.info(f"---node_item_name_confirm---处理结束")
    return state





if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "华为B3-211H显示器好用么？？",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False,default=str))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        logger.exception("==========")
