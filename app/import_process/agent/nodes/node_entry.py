import os
import sys

from pathlib import Path
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    入口节点,决定传入进来的文件走哪一个节点,在此节点中要使用的state的属性是is_read_md_enabled,
    is_read_pdf_enabled,md_path pdf_path file_title
    1.进入节点的日志输出
    2.参数解析
    3.解析文件类型
    4.完成节点的日志输出
    :param state:
    :return:
    """
    #function_name是节点名
    function_name= sys._getframe().f_code.co_name
    # 进入节点的日志输出
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    #返回前端
    add_running_task(state["task_id"],function_name)
    #进行非空测试
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.error(f">>> [{function_name}] 错误：无文件输入请指定要处理的文件路径！")
        return state
    #判定并完成state赋值
    if local_file_path.endswith('.md'):
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
    elif local_file_path.endswith('.pdf'):
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    else:
        logger.error(f">>> [{function_name}] 错误：不支持的文件类型！请检查文件后缀名是否为.md或.pdf")
    #提取出文件名->为了大模型识别item_name
    #两种方式提取文件名
    file_name_os = os.path.basename(local_file_path).split('.')[0]
    file_name= Path(file_name_os).stem
    state["file_title"] = file_name
    #结束
    logger.info(f">>> [{function_name}]结束执行了！现在的状态为：{state}")
    add_done_task(state["task_id"],function_name)
    return state



































