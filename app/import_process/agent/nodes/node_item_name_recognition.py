import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

def step_1_get_chunks(state):
    """
    校验和取值
    :param state:
    :return:取得切出来的块和文件标题
    """
    chunks= state.get("chunks")
    file_title = state.get("file_title")
    if not chunks:
        logger.error(f">>> 接受没有获取到切出来的块！请检查！")
        raise ValueError("chunks没有值，无法继续进行，抛出异常处理！")
    if not file_title:
        logger.error(f">>> 获取文件标题失败！请检查！")
        raise ValueError("file_title没有值，无法继续进行，抛出异常处理！")
    return chunks, file_title

def step_2_build_context(chunks):
    """
    构建上下文
    根据chunks切面的content内容进行分拼接,只要分数前几个(5) ,格式如此:切片：{1}，标题:{title},内容：{content} \n\n
    :param chunks:
    :return:
    """
    #前置准备工作
    parts=[]
    total_chars=0
    #循环处理chunks
    for index,chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
    #转换结果
        chunk_title = chunk.get("title")
        chunk_content = chunk.get("content")
        # if len(chunk_content) + total_chars > SINGLE_CHUNK_CONTENT_MAX_LEN:
        #     chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN-total_chars]
        data = f"切片：{index}，标题:{chunk_title},内容：{chunk_content} \n\n"
        parts.append(data)
        total_chars += len(chunk_content)
        if total_chars > CONTEXT_TOTAL_MAX_CHARS:
            logger.info("已经达到最大字符数:{total_chars}，停止拼接！")
            break
    #返回结果
    context= "\n\n".join( parts)
    return context[:SINGLE_CHUNK_CONTENT_MAX_LEN]


def step_3_call_llm(context, file_title):
    """
    调用大模型 获得item_name
    :param context:
    :param file_title:
    :return:
    """
    # 1. 构建提示词
    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
    system_prompt = load_prompt("product_recognition_system")
    # 2. 获取模型对象
    llm = get_llm_client(json_mode=False)
    # 3. 执行调用
    messages = [
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt)
    ]
    response = llm.invoke(messages)
    # 4. 阶段判断和兜底！
    item_name = response.content
    if not item_name:
        item_name = file_title
    # 5. 返回结果
    return item_name


def step_4_update_chunks_and_state(state, item_name, chunks):
    """
    对state里的item_name属性赋值
    :param state:
    :param item_name:
    :param chunks:
    :return:
    """
    state['item_name'] = item_name

    for chunk in chunks:
        chunk['item_name'] = item_name
    state['chunks'] = chunks
    logger.info(f"完成了chunks和state[item_name]的赋值和修改！！")


def step_5_generate_embeddings(item_name):
    """
    根据item_name生成向量 -》 稠密 + 稀疏
    :param item_name:
    :return:dense_vector [稠密] ,  sparse_vector [稀疏]
    """
    result = generate_embeddings([item_name])
    dense_vector, sparse_vector = result['dense'][0], result['sparse'][0]
    return dense_vector, sparse_vector


def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    """
    将向量和对应字段存入向量数据库
    导入的参数都是向量数据库中集合的属性
    :param file_title:
    :param item_name:
    :param dense_vector:
    :param sparse_vector:
    :return:
        """
    #1.向量数据库客户端
    milvus_client = get_milvus_client()
    #2.检查集合是否为空
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        #创建集合
        # 3.1. 创建集合对应的列的信息
        schema = milvus_client.create_schema(
            auto_id=True,  # 主键自增长
            enable_dynamic_field=True,  # 动态字段
        )

        # 3.2. Add fields to schema
        # pk file_title item_name dense_vector sparse_vector
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 3.3 查询快，配置索引
        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",  # 给哪个列创建索引 稠密
            index_name="dense_vector_index",  # 索引的名字
            index_type="HNSW",  # 配置查找所用的算法
            metric_type="COSINE",  # 配置向量匹配和对比的 IP COSINE
            params={"M": 16,  # Maximum number of neighbors each node can connect to in the graph
                    "efConstruction": 200},  # or "DAAT_WAND" or "TAAT_NAIVE"
        )
        #为稀疏向量建立索引
        index_params.add_index(
            field_name="sparse_vector",  # Name of the vector field to be indexed
            index_type="SPARSE_INVERTED_INDEX",  # Type of the index to create
            index_name="sparse_vector_index",  # Name of the index to create
            metric_type="IP",  # Metric type used to measure similarity
            # 只计算可能得高分的向量，跳过大量的 0
            params={"inverted_index_algo": "DAAT_MAXSCORE"},  # Algorithm used for building and querying the index
        )
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,  # 字段
            index_params=index_params  # 索引
        )
    # 3. 先删除之前存在的item_name
    # 加载和选中集合
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    milvus_client.delete(collection_name=milvus_config.item_name_collection,
                         filter=f"item_name=='{item_name}'")
    # 4. 向集合插入最新的item_name数据和对应的向量即可
    item = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector
    }
    milvus_client.insert(collection_name=milvus_config.item_name_collection,
                         data=[item])
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    logger.info(f"保存了item_name:{item_name}的数据到向量数据库中！！")


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    # 1. 进入的日志和任务状态的配置
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state['task_id'], function_name)
    try:
        # 1. 验和取值 （file_title,chunks）
        # 获取前置的材料！ file_title = 为了兜底，没有item_name
        chunks, file_title = step_1_get_chunks(state)
        # 2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本
        context = step_2_build_context(chunks)
        # 3. 调用模型，拼接提示词，识别chunks对应item_name
        item_name = step_3_call_llm(context, file_title)
        # 4. 修改state chunks -》 item_name -> chunks [{title parent_title context part item_name [没有值]}]
        step_4_update_chunks_and_state(state, item_name, chunks)
        # 5. item_name生成向量（稠密/稀疏）
        dense_vector, sparse_vector = step_5_generate_embeddings(item_name)
        # 6. 将向量存储到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
        step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)
    except Exception as e:
        # 处理异常
        logger.error(f">>> [{function_name}]主体识别发生了异常，异常信息：{e}")
        raise  # 终止工作流
    finally:
        # 6. 结束的日志和任务状态的配置
        logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state}")
        add_done_task(state['task_id'], function_name)
    return state



def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()