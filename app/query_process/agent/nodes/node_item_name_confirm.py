import json
import time
import sys
from app.core.logger import logger
from langchain_core.messages import HumanMessage

from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.core.load_prompt import load_prompt
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
                         f"关联主体： {','.join(history_message['item_names'])},"
                         f"时间： {history_message['ts']}\n")
    prompt= load_prompt("rewritten_query_and_itemnames" ,history_text=history_text,
                        original_query=original_query )
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
    if "item_name" not in dict_content:
        dict_content["item_names"] = []
    if "rewritten_query" not in dict_content:
        dict_content["rewritten_query"] = original_query
    #4.返回数据
    logger.info(f"已经完成问题的重写和item_name的提取！ 结果为：{dict_content}")
    return dict_content

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
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    #1.获取历史消息
    history = get_recent_messages(state["session_id"])
    # 2.保存本次聊天消息
    message= save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", [])
    )
    #3.使用大模型提取item_name并且重写问题
    item_names_and_rewritten_query= step_3_llm_item_name_and_rewrite_query(state["original_query"],history)
    # 后面会调用大模型，进行逻辑处理
    time.sleep(1)
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    print(f"---node_item_name_confirm---处理结束")

    return {"item_names": ["示例商品"]}