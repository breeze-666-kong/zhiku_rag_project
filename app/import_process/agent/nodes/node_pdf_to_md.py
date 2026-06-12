import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests
from sqlalchemy.testing.util import function_named

from app.conf.mineru_config import mineru_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.path_util import PROJECT_ROOT


def step_1_validate_paths(state):
    """
    进行路径校验！ pdf_path失效 直接异常处理!
                local_dir 没有，给与默认值
    :param state:
    :return:
    """
    logger.debug(f">>> [step_1_validate_paths]在md转pdf下，开始进行文件格式校验！！")
    pdf_path= state["pdf_path"]
    local_dir= state["local_dir"]
    #非空判断
    if not pdf_path:
        logger.error(f">>> [step_1_validate_paths] 错误：无文件输入请指定要处理文件路径！")
        raise ValueError(">>> [step_1_validate_paths] 错误:无文件输入请指定要处理文件路径！")
    if not local_dir:
        #不存在,创建一个默认值
        local_dir=str(PROJECT_ROOT/"output")
        logger.info(f">>> [step_1_validate_paths] 提示：未指定输出目录，已使用默认目录：{local_dir}")
    pdf_path_obj= Path(pdf_path)
    local_dir_obj= Path(local_dir)
    if not pdf_path_obj.exists():
        logger.error(f">>> [step_1_validate_paths] 错误：指定的文件不存在！请检查文件路径是否正确！")
        raise FileNotFoundError(f">>> [step_1_validate_paths] 错误：指定的文件不存在！请检查文件路径是否正确！")
    if not local_dir_obj.exists():
        #不存在,创建一个默认值
        local_dir_obj.mkdir(parents=True, exist_ok=True)
        logger.info(f">>> [step_1_validate_paths] 提示：指定的输出目录不存在，已创建目录：{local_dir_obj}")
    return pdf_path_obj, local_dir_obj

def step_2_upload_and_poll(pdf_path_obj):
    """
    将pdf文件使用minerU解析，并且获取md对应的下载的url地址！！
    :param pdf_path_obj:上传解析pdf文件的 path对象
    :return:url , minerU解析后md文件zip压缩包的下载地址
    """
    #申请上传地址解析
    #准备token和url 固定格式的请求头,格式在官网查看
    token= mineru_config.api_key
    url= f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": "vlm"
    }
    response = requests.post(url, headers=header, json=data)
    #结果处理:如果状态码不是200 or 返回结果的状态码不是0,请求失败
    if response.status_code !=200:
        logger.error(f">>> [step_2_upload_and_poll] 错误：请求minerU失败！请检查文件路径是否正确,错误信息{response.text}")
        raise RuntimeError(f">>> [step_2_upload_and_poll] 错误：请求minerU失败！请检查文件路径是否正确,错误信息{response.text}")
    uploaded_url = response.json()['data']['file_urls'][0]  # 上这个地址上传文件
    batch_id = response.json()['data']['batch_id']  # 处理id，后续根据这个id获取结果！

    #将文件上传到对应解析地址
    """
    因为要使用put请求,所以要禁止使用代理,防止出现错误
    """
    http_session= requests.Session()
    http_session.trust_env = False
    try:
        with open(pdf_path_obj, 'rb') as f:
            file_data= f.read()
        uploaded_response= http_session.put(uploaded_url, data=file_data)
        if uploaded_response.status_code != 200:
            logger.error(f"[step_2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确！！")
            raise RuntimeError(f"[step_2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确！！")
    except Exception as e:
        logger.error(f">>> [step_2_upload_and_poll] 错误：上传文件失败！请检查文件路径是否正确,错误信息{e}")
        raise RuntimeError(f"[step_2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确！！")
    finally:
        http_session.close()

    #轮询获得解析结果
    # 循环获取！确保获取到结果，再先后执行！！
    # 设计一个循环，3秒获取一次！ 最多等待10分钟600 -> 600页pdf
    url= f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_second= 600     # 1s -> 1页pdf
    poll_interval= 3    #间隔时间是3秒
    start_time= time.time()
    while True:
        #超时判断
        if time.time() - start_time > timeout_second:
            logger.error(f">>> [step_2_upload_and_poll] 错误：解析pdf超时！请检查文件路径是否正确,错误信息{uploaded_response.text}")
            raise RuntimeError(f">>> [step_2_upload_and_poll] 错误：解析pdf超时！请检查文件路径是否正确,错误信息{uploaded_response.text}")
        #向指定url地址获取解析结果
        res= requests.get(url,headers= header)
       #解析结果并判断,获取zip.url
        if res.status_code != 200:
            #对5xx的状态码再给一次机会,因为5xx是服务器问题,而服务器不是本地的,所以可以再尝试
            if res.status_code >= 500:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口失败，返回的状态码{res.status_code}！！")
        json_data = res.json()  # 获取本次结果
        if json_data['code'] != 0:
            # ！= 0 很大概率token过期了，后续没有钱了！
            raise RuntimeError(
                f"[step_2_upload_and_poll]请求minerU解析接口失败，返回的错误:{json_data['code']}信息{json_data['msg']}！！")

        # 判断下解析状态
        extract_result = json_data['data']['extract_result'][0]
        if extract_result['state'] == 'done':
            # 解析完毕可以获取结果
            full_zip_url = extract_result['full_zip_url']
            logger.info(f"已经完成pdf的解析，耗时：{time.time() - start_time}s,解析结果：{full_zip_url}")
            return full_zip_url
        else:
            # 还没有解析完
            time.sleep(poll_interval)


def step_3_download_and_extract(zip_url, local_dir_obj, stem)->str:
    """
    下载指定的md.zip文件，并且解压，返回解压后的md文件的地址！
    :param zip_url: 要下载的地址
    :param local_dir_obj:存储的文件夹
    :param stem:pdf的文件名字
    :return:返回md文件的地址
    """
    #下载zip文件
    response= requests.get(zip_url)

    if response.status_code != 200:
        logger.error(f">>> [step_3_download_and_extract] 错误：下载文件失败！请检查文件路径是否正确,错误信息{response.text}")
        raise RuntimeError(f">>> [step_3_download_and_extract] 错误：下载文件失败！请检查文件路径是否正确,错误信息{response.text}")
    #下载到本地
    zip_save_path=local_dir_obj/f"{stem}_result.zip"
    with open(zip_save_path, 'wb')as f :
        # response.content 响应体中的数据
        f.write(response.content)
    logger.info(f"[step_3_download_and_extract]下载zip文件成功，保存路径：{zip_save_path}")
    #清空旧目录
    extract_target_dir= local_dir_obj/stem
    if extract_target_dir.exists():
        # 递归进行目录内容删除/本身也会被删除
        shutil.rmtree(extract_target_dir)
    # 创建一个新的目录
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    #进行解压 , 使用工具zipfile,zipfile处理zip文件的模块
    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        # 调用对象的解压方法进行解压即可
        zip_ref.extractall(extract_target_dir)
    #返回md文件的url
    #解压后的文件可能叫文件名,也可能叫full.md
    md_file_list= list(extract_target_dir.rglob("*.md"))#解压得到后的文件名

    if not md_file_list:
        logger.error(f"[step_3_download_and_extract]没有找到md文件，请检查输入文件路径是否正确！！")
        raise RuntimeError(f"[step_3_download_and_extract]没有找到md文件，请检查输入文件路径是否正确！！")

    target_md_file= None  #  存储最终md文件
    for md_file in md_file_list:
        # stem 文件名
        if md_file.name == stem + ".md":
            target_md_file = md_file
            break
        # 检查有没有full.md (第一次没有找到，才找full)
    if not target_md_file:
        for md_file in md_file_list:
            # stem 文件名
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break
        # 实在没有我就获取第一个就行
    if not target_md_file:
        target_md_file = md_file_list[0]

    #重命名操作
    if target_md_file.stem != stem:
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

    final_md_str_path = str(target_md_file.resolve())
    logger.info(f"[step_3_download_and_extract]完成md解压，最终存储md路径为：{final_md_str_path}")
    return final_md_str_path

def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    将pdf转成md节点,核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    1.进入日志和任务状态的配置
    2.进行参数校验
    3. 调用minerU进行pdf的解析（local_file_path）返回一个下载文件的地址 xx.zip url地址
        4. 下载zip包，并且解析和提取 （local_dir）
        5. 把md_path地址进行赋值，读取md的文件内容 md_content赋值（文本内容）
        6. 结束的日志和任务状态的配置
    :param state:
    :return:
    """
    #返回前端
    function_name= sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state["task_id"],function_name)
    try:
        #进行参数校验
        pdf_path_obj,local_dir_obj= step_1_validate_paths(state)
        #调用minerU进行pdf的解析
        zip_url= step_2_upload_and_poll(pdf_path_obj)
        #下载zip包，并且解析和提取 （local_dir）,返回值：解压后md文件的真实路径
        md_path= step_3_download_and_extract(zip_url,local_dir_obj,pdf_path_obj.stem)
        #更新数据
        state['md_path'] = md_path
        state['local_dir'] = local_dir_obj  # 主要处理下！是str类型
        # md的内容读取，配置给md_content
        with open(md_path, 'r', encoding='utf-8') as f:
            state['md_content'] = f.read()
    except Exception as e:
        logger.error(f">>> [{function_name}] 错误：PDF转MD失败！错误信息{e}")
        raise
    finally:
        #结束执行
        logger.info(f">>> [{function_name}]结束执行了！现在的状态为：{state}")
        add_done_task(state["task_id"],function_name)
    return state



if __name__ == "__main__":
    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")


































