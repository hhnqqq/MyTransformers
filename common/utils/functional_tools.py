"""
@function:一些有用的小工具
@author:hhn
@time:2024-01
"""
import sys
import time
import re
import os
import warnings
import datetime
import inspect

import logging
import typing
import pkg_resources
import traceback

from inspect import signature
from functools import wraps, partial

def print_progress_bar(iteration=None, 
                       total=None, 
                       prefix='', 
                       suffix='', 
                       decimals=1, 
                       length=50, 
                       fill='-', 
                       arrow='>',
                       print_end="\r"):
    """
    作用：打印进度条的函数
    输入：当前伦次，总轮次，前后缀，进度条长度，进度条字符
    输出：带前后缀的进度条"""
    assert iteration is not None or total is not None, '当前轮次和总轮次不能为空'
    if prefix == '' and suffix == '':
        warnings.warn('当前的进度条前后缀为空，请确保不需要前后缀')
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * (filled_length-1) + arrow + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    
    if iteration == total:
        print()

def attach_wrapper(obj, func=None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func

def timer(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def retry(max_attempts:int=3, delay:int=1, print_trace_back=False, return_error_info=False):
    """重试装饰器
    给装饰器再加一层，可以实现添加参数的效果
    带参数的装饰器如果直接用@retry会出错，而通过partial可以时装饰器不通过@retry()方式调用也可以
    [因为直接使用时max_attempts和delay必须被关键字指定]
    """
    assert isinstance(max_attempts, int) and isinstance(delay, int), '参数必须是整数'

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if print_trace_back:
                        e = traceback.format_exc()
                        error_info = f">>>函数{func.__name__}第{attempts + 1}次尝试失败，报错信息为: {e}"
                        print(error_info)
                    time.sleep(delay)
                    attempts += 1
            if return_error_info:
                return error_info
            else:
                return None
        
        @attach_wrapper(wrapper)
        def set_max_attempts(new_max_attempts):
            nonlocal max_attempts
            max_attempts = new_max_attempts

        @attach_wrapper(wrapper)
        def set_delay(new_delay):
            nonlocal delay
            delay = new_delay

        wrapper.get_attempts = lambda: max_attempts
        wrapper.get_delay = lambda: delay
        return wrapper
    return decorator

def auto_logging(level, name=None, message=None):
    """
    自动记录日志
    可以通过set_level和set_message改变日志记录的内容
    """
    def decorator(func):
        log_name = name if name else func.__module__
        logger = logging.getLogger(log_name)
        log_message = message if message else func.__name__
    
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(level, log_message)
            return func(*args,**kwargs)
        
        @attach_wrapper(wrapper)
        def set_level(new_level):
            '''通过attach_wrapper装饰器，
            让set_level()函数成为wrapper装饰器的属性wrapper.set_level，
            可以通过func.set_level(new_level)使用'''
            nonlocal level
            level = new_level

        @attach_wrapper(wrapper)
        def set_message(new_message):
            nonlocal message
            message = new_message

        wrapper.get_level = lambda: level
        wrapper.get_message = lambda: message
        return wrapper
    return decorator
        
def type_assert(*type_args, **type_kwargs):
    def decorator(func):
        if not __debug__:
            return func
        
        sig = signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.apply_defaults.items():
                if name in bound_types and not isinstance(value, bound_types[name]):
                    raise TypeError(f'参数{name}必须是{bound_types[name]}类型')
            return func(*args, **kwargs)
        return wrapper
    return decorator        

def ensure_directory_exists(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('>>> 目录不存在，已新建对应目录')

def print_separator(char='-'):
    """打印分隔线"""
    print(char * 80)

def get_current_time() -> str:
    """获取当前时间的字符串表示"""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def ask_yes_no(question):
    """询问Yes/No问题"""
    answer = input(f"{question} (y/n): ").lower()
    while answer not in ('y', 'n'):
        answer = input("Please enter 'y' or 'n': ").lower()
    return answer == 'y'

def debugger():
    '''作用：命令行debug环境控制器
    使用方式：在代码开头设置os.environ(["debug"])=True
    from utils import debugger
    debugger()
    code'''
    import pdb
    try:
        is_debug = eval(os.environ.get("debug"))
    except:
        is_debug = False
    if is_debug:
        d = pdb.Pdb()
        d.set_trace(sys._getframe().f_back)

def re_search(regex: typing.Union[typing.Pattern[str], str],
              text: typing.AnyStr,
              dotall: bool=True,
              default: str="") -> str:
    """
    抽取正则规则的结果

    :param regex: 正则对象或字符串
    :param text: 被查找的字符串
    :param dotall: 正则.是否匹配所有字符
    :param default: 找不到时的默认值
    :return: 抽取正则规则的结果
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if not isinstance(regex, list):
        regex = [regex]
    for rex in regex:
        rex = (re.compile(rex, re.DOTALL)
               if dotall else re.compile(rex)) if isinstance(rex, str) else rex
        match_obj = rex.search(text)
        if match_obj is not None:
            t = match_obj.group().replace('\n', '')
            return t
    return default

def get_package_version(package_name):
    try:
        version = pkg_resources.get_distribution(package_name).version
        return version
    except pkg_resources.DistributionNotFound:
        return None
    
def has_parameter(func, parameter_name):
    """
    判断函数是否具有特定参数
    :param func: 要检查的函数
    :param parameter_name: 要检查的参数名
    :return: True（如果函数具有该参数）或 False（如果函数不具有该参数）
    """
    signature = inspect.signature(func)
    parameters = signature.parameters
    return parameter_name in parameters

def sort_files_by_mtime(directory):
    """
    Sort files in a directory by their modification time, with the most recent first.

    Parameters:
    directory (str): The path to the directory containing the files to sort.

    Returns:
    list: A list of file paths sorted by their modification time, most recent first.
    """
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    files_with_mtime = [(f, os.path.getmtime(os.path.join(directory, f))) for f in files]
    files_with_mtime.sort(key=lambda x: x[1], reverse=True)
    sorted_files = [f[0] for f in files_with_mtime]
    return sorted_files

def open_file_dialog():
    layout = [
        [sg.Text("Choose a pdf file"), sg.Input(), sg.FileBrowse()],
        [sg.Button("OK"), sg.Button("Cancel")]
    ]
    window = sg.Window("File Browser", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Cancel":
            window.close()
            return None
        if event == "OK":
            window.close()
            return values[0]


def manage_sys_prompt(personal_info, sys_prompt, file_path=None, info_before_sys=False):
    if personal_info is None:
        return sys_prompt
    assert (isinstance(personal_info, dict))
    
    if any(info is not None for info in personal_info.values()):
        sys_prompt_prefix = "please remenber below informations for the follow chat: "
        for k, v in personal_info.items():
            if isinstance(v, int):
                v = str(v)
            if v is not None and v != "":
                sys_prompt_prefix += 'my ' + k + 'is: ' + v + '\n'
        sys_prompt_prefix += '\n'

    if file_path:
        import fitz
        document = fitz.open(file_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        read_prompt = "please remenber below file content for the follow chat: " + text
    else:
        read_prompt = ""

    if info_before_sys:
        return sys_prompt_prefix + sys_prompt + read_prompt
    else:
        return sys_prompt + sys_prompt_prefix + read_prompt
