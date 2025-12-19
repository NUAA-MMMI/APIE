import json
import re
import random

from jsonschema import validate

from setting.loggerConfig import getLogger
import numpy as np
logger = getLogger()

def getRequiredTasks(schema : dict) -> tuple:

    entityTask = False
    relationTask = False
    eventTask = False

    try:
        tasks = schema['tasks']
        if 'NER' in tasks:
            entityTask = True
        if 'RE' in tasks:
            relationTask = True
        if 'EE' in tasks:
            eventTask = True

    except KeyError:
        logger.error('schema does not have required key')
        return entityTask, relationTask, eventTask
    
    return entityTask, relationTask, eventTask


def getSampleList(samplePoll:list, method:str, **kwargs) -> list:
    # Get the sample list
    sampleList = []

    #* Zero-shot learning
    if method == "zsl": 
        pass
    #* Random-shot learning
    elif method == "rsl": 
        shotSize = kwargs['shotSize']
        sampleList = random.sample(samplePoll, shotSize)
    #* Act-UIE learning
    elif method == "actuie":
        pass

    return sampleList

def getInstruction(schema : dict, method : str='zsl', sampleList = []) -> dict:
    # Get the required tasks from the schema
    entityTask, relationTask, eventTask = getRequiredTasks(schema)

    # Get the Language from the schema
    language = schema['language'] if 'language' in schema else 'en'

    # Get the schema properties
    properties = schema['schema'] if 'schema' in schema else {}

    if language == 'en':
        instruction = ""
        if method != "onlysample":
        # English instruction
            instruction = "You are an expert in information extraction. Please extract the types that match the pattern from the input."
            instruction += "Please reply in JSON string format:\n"
            # if method == "zsl":
            #     instruction += "'''json\n"
            #     instruction += json.dumps(schema['ZSL'], indent=2, ensure_ascii=False) + '\n'
            #     instruction += "'''\n"
            instruction += "'''json\n"
            instruction += json.dumps(schema['ZSL'], indent=2, ensure_ascii=False) + '\n'
            instruction += "'''\n"
            instruction += "If it does not exist, an empty list is returned."

            if entityTask == True:
                instruction += f"The possible types of the \"entity\" is: {schema['classes'][0]}\n"
            if relationTask == True:
                instruction += f"The possible types of the \"relation\" is: {schema['classes'][1]}\n"
            if eventTask == True:
                instruction += f"The possible types of the \"event\" is: {schema['classes'][2]}\n"

        if len(sampleList) > 0:
            instruction +=  f"\nSpecific input-output examples are as follows:\n"
            for sample in sampleList:
                instruction += f"Input: {sample['text']}\n"
                instruction += f"Output:'''json {json.dumps(sample['standard'], indent=2, ensure_ascii=False)}'''\n"
    elif language == 'zh':
        # Chinese instruction
        instruction = "你是一个信息抽取专家。请从输入中提取符合模式的类型。请以JSON字符串格式回复：\n"
        instruction += json.dumps(schema['ZSL']+'\n', indent=2, ensure_ascii=False)

        if entityTask == True:
            instruction += f"\"entity\"的可能类型是: {schema['classes'][0]}\n"
        if relationTask == True:
            instruction += f"\"relation\"的可能类型是: {schema['classes'][1]}\n"
        if eventTask == True:
            instruction += f"\"event\"的可能类型是: {schema['classes'][2]}\n"

        if len(sampleList) > 0:
            instruction +=  f"\n具体的输入输出示例如下：\n"
            for sample in sampleList:
                instruction += f"输入: {sample['text']}\n"
                instruction += f"输出: {json.dumps(sample['standard'], indent=2, ensure_ascii=False)}\n"
    # Remove any extra spaces and newlines
    instruction = re.sub(r'\s+', ' ', instruction).strip()

    return instruction

def getJsonFormat(origionResponse : str ,schema : dict) :

    logger = getLogger("getJsonFormat")

    JsonResponse = None

    match = re.search(r'(?:```json\s*([\s\S]*?)\s*```)|(?:\'\'\'json\s*([\s\S]*?)\s*\'\'\')', origionResponse)

    if match:
        JsonResponse = match.group(1) or match.group(2)
        try:
            JsonResponse = json.loads(JsonResponse)
        except:
            logger.debug(f"\nOrigion Response: {origionResponse}\n Inhand Response: {JsonResponse}")
            JsonResponse = None
            return JsonResponse
    else:
        logger.debug(f"\nOrigion Response: {origionResponse}\n Inhand Response: {JsonResponse}")
        JsonResponse = None
        return JsonResponse

    return JsonResponse

def validateJsonFormat(JsonResponse : dict, schema : dict) -> bool:
    logger = getLogger("validateJsonFormat")

    # Validate the JSON response against the schema
    try:
        validate(instance=JsonResponse, schema=schema['schema'])
        return True
    except Exception as e:
        logger.error(f"JSON validation error: {e}")
        return False
    
def dict2tuple(element):
    """
    递归函数，将嵌套字典和列表转换为元组。
    """
    if isinstance(element, dict):
        # 将字典的键值对排序并递归转换为元组
        return tuple(sorted((k, dict2tuple(v)) for k, v in element.items()))
    elif isinstance(element, list):
        # 将列表中的元素递归转换为元组
        return tuple(dict2tuple(v) for v in element)
    else:
        # 基本类型直接返回
        return element

def list2set(inputList):
    """
    将一个元素为字典的列表递归地转化为集合。
    """
    # 将列表中的每个字典递归转换为元组，然后创建集合
    return set(dict2tuple(d) for d in inputList)

def indicator(std: set, pred: set):
    """
    Calculate the number of true positives, false positives, false negatives, and true negatives.

    Args:
    - std (set): The standard set, representing the ground truth.
    - pred (set): The predicted set.

    Returns:
    - TP (int): True Positives count, items correctly predicted in both sets.
    - FP (int): False Positives count, items predicted but not in the standard set.
    - FN (int): False Negatives count, items in the standard set but not predicted.
    - TN (int): True Negatives count, always zero in this context.
    """
    TP = len(std & pred)  # True Positives
    FP = len(pred - std)  # False Positives
    FN = len(std - pred)  # False Negatives
    TN = 0  # True Negatives (not applicable)

    return TP, FP, FN, TN

def similarity(LL: list[set]) -> float:
    """
    Calculate the similarity between a list of sets.

    The similarity is calculated as the size of the intersection divided by the size of the union.

    Args:
    - LL (list[set]): A list of sets.

    Returns:
    - float: The similarity between the sets, ranging from 0 to 1.
    """
    if len(LL) == 0:
        # If the list is empty, return 0
        return 0
    
    intersection = set.intersection(*LL)
    logger.debug(f"Intersection: {intersection}")
    union = set.union(*LL)
    logger.debug(f"Union: {union}")
    return len(intersection) / len(union) if len(union) > 0 else 1

def precision(TP: int, FP: int) -> float:
    """
    Calculate the precision given the True Positives and False Positives counts.

    The precision is the ratio of True Positives to the sum of True Positives and False Positives.

    Args:
    - TP (int): True Positives count.
    - FP (int): False Positives count.

    Returns:
    - float: The precision, ranging from 0 to 1.
    """
    if TP + FP == 0:
        return 1
    else:
        return TP / (TP + FP)
def recall(TP, FN):
    """
    Calculate the recall given the True Positives and False Negatives counts.

    Recall is the ratio of True Positives to the sum of True Positives and False Negatives.

    Args:
    - TP (int): True Positives count.
    - FN (int): False Negatives count.

    Returns:
    - float: The recall, ranging from 0 to 1.
    """
    if TP + FN == 0:
        # Avoid division by zero; recall is undefined in this case
        return 1
    else:
        return TP / (TP + FN)
def f1score(TP, FP, FN):
    """
    Calculate the F1-score given the True Positives, False Positives and False Negatives counts.

    The F1-score is the harmonic mean of precision and recall, ranging from 0 to 1.

    Formula: F1 = 2 * TP / (2 * TP + FP + FN)

    Args:
    - TP (int): True Positives count.
    - FP (int): False Positives count.
    - FN (int): False Negatives count.

    Returns:
    - float: The F1-score, ranging from 0 to 1.
    """
    if TP + FP + FN == 0:
        # Avoid division by zero; F1-score is undefined in this case
        return 1
    else:
        return 2 * TP / (2 * TP + FP + FN)

def accuracy(TP: int, FP: int, FN: int, TN: int) -> float:
    """
    Calculate the accuracy given the True Positives, False Positives, False Negatives, and True Negatives counts.

    Accuracy is the ratio of correct predictions to all predictions.

    Formula: Accuracy = (TP + TN) / (TP + FP + FN + TN)

    Args:
        TP (int): True Positives count.
        FP (int): False Positives count.
        FN (int): False Negatives count.
        TN (int): True Negatives count.

    Returns:
        float: The accuracy, ranging from 0 to 1.
    """
    if TP + FP + FN + TN == 0:
        # Avoid division by zero; accuracy is undefined in this case
        return 1
    else:
        return (TP + TN) / (TP + FP + FN + TN)

def normalizeData(data):
    """
    Normalize numerical data in a list or a numpy array to the range [0, 1].

    Args:
        data (list or numpy.ndarray): The data to normalize.

    Returns:
        list or numpy.ndarray: Normalized data in the same type as input.
    """
    try:
        arr = np.array(data, dtype=float)
        min_val = arr.min()
        max_val = arr.max()
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    except ImportError:
        # Fallback to pure Python if numpy is not available
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            return [0 for _ in data]
        return [(x - min_val) / (max_val - min_val) for x in data]
    
def getELD(data: dict) -> float:
    length = len(data['text'])
    count = 0
    if 'entities' in data['standard']:
        count += len(data['standard']['entities'])
    if 'relations' in data['standard']:
        count += len(data['standard']['relations'])
    if 'events' in data['standard']:
        count += len(data['standard']['events'])
    return count/length if length > 0 else 0

import numpy as np
from nltk.metrics import edit_distance
def getTextUncertainty(texts: list[str]) -> float:
    """
    Calculate the uncertainty of a list of texts based on their length.

    Args:
        texts (list[str]): A list of text strings.

    Returns:
        float: The average uncertainty based on the length of the texts.
    """
    # 计算所有两两编辑距离
    distances = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            d = edit_distance(texts[i], texts[j])
            distances.append(d)

    uncertainty = np.mean(distances)
    return uncertainty if len(distances) > 0 else 0.0
