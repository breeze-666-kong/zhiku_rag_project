import json
import os
import sys
import re
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from langchain_text_splitters import RecursiveCharacterTextSplitter





#============================    参数配置    =================================
DEFAULT_MAX_CONTENT_LENGTH = 2000 # 512 - 1500 token,最大块的长度
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500 # 最小的长度


def step_1_get_content(state):
    """
    获取参数内容,md_content, file_title
    :param state:
    :return:md文本内容,文件名
    """
    md_content= state["md_content"]
    if not md_content:
        logger.error(f">>> [step_1_get_content]没有有效的md内容，直接抛出异常！！！！")
        raise Exception("md_content参数为空")
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')
    file_title = state.get("file_title", "default_file")
    return md_content, file_title





def step_2_split_by_title(md_content, file_title):
    """
    先对文档进行粗处理,按照标题切分,来进行语义切割
    注意:在md文档中,会以####来作为标题分割,但要注意代码块,代码块通常以```和```作为开头结尾
    :param md_content:
    :param file_title:
    :return:[{content,title,file_title}]
    """
    #正则,来判断标题
    title_pattern = r'^\s*#{1,6}\s+.+'
    #把文档按照每行进行切分,方便接下来的操作
    lines= md_content.split('\n')
    current_title=""
    current_lines=[]#当前标题行
    title_count=0#标题出现次数
    is_code_block= False#是否为代码块
    sections=[]#返回的切割结果 格式[{content,title,file_title}]
    for line in lines:
        strip_line= line.strip()#为了去除md中每一行的空格
        # 判断是否为代码块,使用strip_line是为了避免误判
        if strip_line.startswith("```") or strip_line.startswith('~~~'):
            is_code_block= not is_code_block
            #如果是代码块的开头和结尾就把内容存进去
            current_lines.append(line)
            continue
        # 判断是否为标题,使用正则和不是代码块来确定
        is_title= (not is_code_block) and re.match(title_pattern, strip_line)
        if is_title:
            #在循环当中把每一次的内容存进要返回的内容中去,后续写覆盖内容的代码
            if current_title:
                sections.append({
                    "title":current_title,
                    "content": "\n".join(current_lines),
                    "file_title":file_title})
            current_title= strip_line
            title_count += 1
            current_lines = [current_title]
        else:
            current_lines.append(line)

    #保存最后一个
    if current_title:
        sections.append( {
            "title":current_title,
            "content": "\n".join(current_lines),
            "file_title":file_title
        })
    logger.info(f"已经完成chunks的语义粗切！识别chunk数量：{title_count},切片内容:{sections}")
    return sections, title_count, len(lines)


def split_long_section(section, max_length):
    """
    对于太长的文本进行切割
    :param section:
    :param mex_length:
    :return:
    """
    #1.先获得内容
    content= section.get("content")
    sub_sections = []#定义返回值
    #2.判断长度
    if len(content) <= max_length:
        logger.info(f"[split_long_section]:{content}当前chunk长度小于等于{max_length}，不做二次切割！")
        return [section]
    #3.进行切割
    #创建文本切割器
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=100,
        separators=['\n\n', '\n', '。', '！',"；"," "] #切割的符号（什么节点切割）
    )
    #进行切割,为了返回清楚的返回,enumerate函数中返回的是元组，元组中第一个元素是索引，第二个元素是切割后的内容
    for index,chunk in enumerate(splitter.split_text(content),start=1):
        #去除空白
        sub_content= chunk.strip()
        title= f"{section.get('title')}_{index}"
        file_title= section.get("file_title")
        part= index
        parent_title= section.get("parent_title")
        sub_sections.append({
            "title":title,
            "content":sub_content,
            "file_title":file_title,
            "part":part,
            "parent_title":parent_title
        })
    #4,返回数据
    return sub_sections

def merge_short_sections(final_sections, min_length):
    """
    对上一次切的太碎的合并一下,不得小于最小值,本质逻辑相当于一个双指针,
    如1 2 3 4 5  pre指针从1开始,1太小,就看2里的parent_title是不是同一个,是同一个就合并,然后继续直到不符合两个条件为止
    如果开始不符合就pre指针从1到2,从2开始同样的操作
    :param final_sections: 昨晚长文本切割的列表
    :param min_length:
    :return:
    """
    merged_sections = []   #定义返回值
    pre_section= None   #当前正在处理的块
    for section in final_sections:
        #判断是不是第一次进来
        if pre_section is None:
            pre_section= section
            continue
        #对两个条件进行判断
        #条件1:块小于最小值
        #条件2:是否为同一个父标题
        is_pre_short= len(pre_section.get("content")) < min_length
        is_same_parent= pre_section.get("parent_title") and (pre_section.get("parent_title") == section.get("parent_title"))
        if is_pre_short and is_same_parent:
            #合并
            pre_section["content"] += "\n\n" + section.get("content")
            pre_section["part"] = section.get("part")
        else:
            #不进行合并,但是把pre_section换成现在的section
            pre_section=section
            merged_sections.append(pre_section)
        #对最后一个进行处理
    if pre_section is not None:
        merged_sections.append(pre_section)

    return merged_sections

def step_3_refine_chunks(sections, mex_length, min_length):
    """
    对粗处理完的文本做细处理,也就是对于太长的文本进行切割,对于太短的进行合并
        1. 超过了MIN_CONTENT_LENGTH块，要做切割！ （parent_title | part ）
        2. 小于了MIN_CONTENT_LENGTH块，要合并结果！ （同一个parent_title)
    :param sections:粗处理完的文本列表,里面是字典
    :param mex_length:最大文本块
    :param min_length:最小文本块
    :return:细处理后的文本
    """
    final_sections= []#定义最后输出
    for section in sections:
        #对于太长的文本切碎
        sub_section= split_long_section(section,mex_length)
        #使用extend的目的是添加sub_section的每一个元素
        final_sections.extend(sub_section)
    #对于块太小的,需要再次经行合并
    final_sections= merge_short_sections(final_sections,min_length)
    #对于只有粗处理没有进行细处理的数据,我们要补全参数(part,parent_title)防止在向量数据库中出现错误
    for section in final_sections:
        section["part"]= section.get('part') or 1
        section['parent_title'] = section.get('parent_title') or section.get('title')
    return final_sections


def step_4_backup_chunks(state, sections):
    """
    将切割完的碎片进行存储！！！
    :param state: 本地地址  local_dir
    :param sections: 要存储的内容 [{}]
    :return:
    """
    local_dir = state.get("local_dir")
    backup_file_path = os.path.join(local_dir, "chunks.json")
    with open(backup_file_path, "w",encoding="utf-8") as f:
        json.dump(
            sections,  #将什么数据写到指定的文件流！
            f, # 写出的位置
            ensure_ascii=False, #中文直接原文存储
            indent=4  # json带有缩进 4
        )
    logger.info(f"已经将内容,进行备份到:{backup_file_path}")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    完成md内容的切块！
   最终： chunks -> 存储块的集合   chunks ->  备份到本地 -> chunks.json
   1. 参数校验 （材料是否完整）
   2. 粗粒度切割（md）语义完善 -》 使用标题切割  （保证语义）
   3. 特殊场景，一个文档没有标题，我们给他一个默认标题 （兜底 文档 -》 没有标题 ）
   4. 细粒度切割（md）大小和重叠合适 -> 大 -》（设置重叠） 小 || 小 -》 合并  （大 -》 小 || 小 -》 合并）
      大小合适，语义完整的chunks
   5. 数据的备份和chunks属性的修改 (chunks -> state  | chunks -> 本地备份一下)
   返回 state
    """
    function_name = sys._getframe().f_code.co_name
    add_running_task(state['task_id'], function_name)
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    try:
        #1.要先获取状态
        md_content, file_title=step_1_get_content(state)
        #2.先粗切一下,也就是按照标题进行切分
        # [{content:标题的内容,title：标题,file_title：文件名},{},{}]
        sections, title_count, lines_count = step_2_split_by_title(md_content, file_title)
        #3如果这个文档没有标题,就给它创造一个标题
        if title_count == 0:
            # 证明没有标题
            sections = [{"title":"没有主题","content":md_content,"file_title":file_title}]
        #4.进行细切割,要保证大小和重叠合适
        sections = step_3_refine_chunks(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
        # 大小合适，语义完整的chunks
        # 5. 数据的备份和chunks属性的修改 (chunks -> state  | chunks -> 本地备份一下)
        state['chunks'] = sections
        step_4_backup_chunks(state, sections)
    except Exception as e:
        # 处理异常
        logger.error(f">>> [{function_name}]使用minerU解析发生了异常，异常信息：{e}")
        raise  # 终止工作流
    finally:
        # 6. 结束的日志和任务状态的配置
        logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state}")
        add_done_task(state['task_id'], function_name)

    return state

if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")