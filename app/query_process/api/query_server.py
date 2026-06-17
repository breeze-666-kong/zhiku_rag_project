# 6个接口  健康状态 返回页面  发起提问   sse长连接  查看历史对话  清空历史对话
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app


#定义fastapi对象
app= FastAPI(title="Query Service",description= "掌柜智库查询服务！")


# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#健康状态
@app.get("/health")
async def health():
    """
    健康状态
    :return:
    """
    logger.info("健康状态检查,没问题")
    return {"status": "ok"}

#返回页面chat.html
@app.get("/chat.html")
async def get_chat_page():
    """
    返回页面chat.html
    :return:
    """
    chat_html_path = PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="页面不存在！！")
    return FileResponse(chat_html_path)



#创建一个接收参数的类
class QueryRequest(BaseModel):
    query: str = Field(..., title="用户问题", description="用户输入的问题")
    session_id:str= Field(..., title="会话ID", description="会话ID,如果没有生成一个uuid")
    is_stream:bool= Field(..., title="是否流式输出", description="是否流式输出")


def run_query_graph(session_id: str, query: str, is_stream: bool):
    update_task_status(session_id, "processing", is_stream)
    state= create_query_default_state(session_id=session_id,
                                      query=query,
                                      is_stream=is_stream)
    try:
        query_app.invoke(state)
        # 本次任务开启了！ is_stream = True 把结果加入到队列，sse可以取到
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        logger.exception(f"---session_id = {session_id},查询流程出现异常！！{str(e)}")
        # 修改 event = process
        update_task_status(session_id, "failed", is_stream)
        # 推送指定类型的事件
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    发起提问
    :param request:请求参数
    :param background_:异步执行函数  is_stream = True
    """
    query=request.query
    session_id=request.session_id or str(uuid.uuid4())
    is_stream=request.is_stream
    if is_stream:
        create_sse_queue(session_id)
        background_tasks.add_task(run_query_graph,session_id, query, is_stream)
        logger.info(f"query:{query}已经开启了异步和流式处理！！")
        return {
            "session_id": session_id,
            "message": "本次查询处理中...."
        }
    else:
        #同步执行
        run_query_graph(session_id,query, is_stream)

        answer= get_task_result(session_id,"answer")
        logger.info(f"---session_id---- = {session_id},同步执行,查询结果：{answer}")
        return {
            "answer": answer,
            "session_id": session_id,
            "message":"本次查询结束",
            "done_list": []
        }
#sse长连接
@app.get("/stream/{session_id}")
async def stream(session_id: str,request: Request):
    """
    sse长连接方法
    :param session_id:
    :param request:前端的原生请求对象，可以判断是否断开连接！！
    :return:
    """
    logger.info(f"---session_id---- = {session_id},sse长连接开始")
    return StreamingResponse(sse_generator(session_id,request),media_type="text/event-stream")

#查看历史对话
@app.get("/history/{session_id}")
async def get_history(session_id: str,limit: int = 10):
    """
    查看历史对话
    :param session_id:
    :return:
    """
    try:
        logger.info(f"---session_id---- = {session_id},查看历史对话开始")
        chats= get_recent_messages(session_id,limit)
        # items = []
        # for chat in chats:
        #     items.append(chat)
        return {
            "session_id":session_id,
            "items":chats
        }
    except Exception as e:
        logger.error(f"查看历史对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查看历史对话失败: {str(e)}")
#清空历史对话
@app.delete("/history/{session_id}")
async def clear_history_endpoint(session_id: str):
    """
    清空历史对话
    :param session_id: 会话ID
    :return: 删除结果
    """
    logger.info(f"---session_id---- = {session_id},清空历史对话开始")
    try:
        deleted_count = clear_history(session_id)
        return {
            "session_id": session_id,
            "deleted_count": deleted_count,
            "message": f"已清空 {deleted_count} 条历史记录"
        }
    except Exception as e:
        logger.error(f"清空历史对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空历史对话失败: {str(e)}")





if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8082)