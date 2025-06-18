from openai import AzureOpenAI  # openai>=1.0.0
import time
import json
import networkx as nx
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import random
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import requests


# Global variable to hold API cost across multiple LLM calls
api_total_cost = 0.0

clients = {
    "gpt-4": {
        'endpoint': "YOUR AZURE ENDPOINT HERE",
        'api_key': "YOUR AZURE API",
        'api_version': "2024-12-01-preview",
        'name': 'gpt-4-1106-preview-nofilter',
        'input_price': 2.75 / 10 ** 6,
        'output_price': 11.0 / 10 ** 6,
        },
    "gpt-4o": {
        'endpoint': "YOUR AZURE ENDPOINT HERE",
        'api_key': "YOUR AZURE API",
        'api_version': "2024-12-01-preview",
        'name': 'gpt-4o-1120-nofilter-global',
        'input_price': 2.75 / 10 ** 6,
        'output_price': 11.0 / 10 ** 6,
        },
    "gpt-3.5-turbo": {
        'endpoint': "YOUR AZURE ENDPOINT HERE",
        'api_key': "YOUR AZURE API",
        'api_version': "2024-12-01-preview",
        'name': 'gpt-35-turbo-1106-nofilter',
        'input_price': 1.5 / 10 ** 6,
        'output_price': 2.0 / 10 ** 6,
        },
    "o1": {
        'endpoint': "YOUR AZURE ENDPOINT HERE",
        'api_key': "YOUR AZURE API",
        'api_version': "2024-12-01-preview",
        'name': 'o1-1217-nofilter',
        'input_price': 16.5 / 10 ** 6,
        'output_price': 66.0 / 10 ** 6,
        },
    "o3-mini": {
        'endpoint': "YOUR AZURE ENDPOINT HERE",
        'api_key': "YOUR AZURE API",
        'api_version': "2024-12-01-preview",
        'name': 'o3-mini-250131-nofilter-global',
        'input_price': 1.21 / 10 ** 6,
        'output_price': 4.84 / 10 ** 6,
        },
    "o1-preview": {
        'endpoint': "YOUR AZURE ENDPOINT HERE",
        'api_key': "YOUR AZURE API",
        'api_version': "2024-12-01-preview",
        'name': 'o1-preview-0912-nofilter',
        'input_price': 16.5 / 10 ** 6,
        'output_price': 66.0 / 10 ** 6,
        },
    "R1": {
        'endpoint': "YOUR AZURE ENDPOINT HERE",
        'api_key': "YOUR AZURE API",
        'api_version': "2024-12-01-preview",
        'name': 'DeepSeek-R1-nofilter',
        'input_price': 0.0 / 10 ** 6,
        'output_price': 0.0 / 10 ** 6,
        },
}

def get_reasoning_steps(reasoning: str) -> List[str]:
    """
    Splits the reasoning into individual steps.
    """
    return [x.strip() for x in reasoning.split(".")]

def get_reference_info(reference: str) -> List[str]:
    """
    Extracts the information used from the reference.
    """
    return [x.strip() for x in reference.split(".")]

def al_mean(alignment_scores) -> float:
        if len(alignment_scores) == 0:
            return 0.0
        return sum(alignment_scores) / len(alignment_scores)

def compute_usage(response, engine):
    usage = response.usage.to_dict()
    input = usage["prompt_tokens"]
    reasoning = 0 if "completion_tokens_details" not in usage else usage["completion_tokens_details"]["reasoning_tokens"]
    output = usage["completion_tokens"] - reasoning

    cost = {
        "input": input * clients[engine]['input_price'],
        "output": output * clients[engine]['output_price'],
    }

    cost["total"] = sum(cost.values())

    return cost

def run_llm(prompt, temperature = 0.0, max_tokens = 3000, engine="gpt-4o", max_attempt = 10):
    global api_total_cost  # declare to modify the global variable
    if engine == 'R1':
        client = ChatCompletionsClient(
            endpoint=clients[engine]['endpoint'],
            credential=AzureKeyCredential(clients[engine]['api_key']),
        )
    else:
        client = AzureOpenAI(
            azure_endpoint=clients[engine]['endpoint'],
            api_key=clients[engine]['api_key'],
            api_version=clients[engine]['api_version']
        )
        
    if engine == "o1-preview":
        messages = []
    else:
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    flag = 0
    while(flag == 0 and max_attempt > 0):
        max_attempt -= 1
        try:
            if engine in {"o1", "o3-mini", "o1-preview"}:
                response = client.chat.completions.create(
                    model=clients[engine]['name'],
                    messages = messages,
                    max_completion_tokens=max_tokens,
                    frequency_penalty=0,)
            elif engine == "gpt-3.5-turbo" or engine == "gpt-4":
                response = client.chat.completions.create(
                    model=clients[engine]['name'],
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens + len(messages[0]['content']),
                    frequency_penalty=0,)
            elif engine == "R1":
                response = client.complete(
                    messages=messages,
                    max_tokens=max_tokens + len(messages[0]['content']),
                    model=clients[engine]['name']
                )
            else:
                response = client.chat.completions.create(
                    model=clients[engine]['name'],
                    messages = messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    frequency_penalty=0,)
            result = response.choices[0].message.content
            if engine != "R1":
                cost = compute_usage(response, engine)
                # Update the global API cost counter
                api_total_cost += cost["total"]
            # print(f"Total API cost so far (accumulated in run_llm calls): {api_total_cost}")
            flag = 1
        except Exception as e:
            print(e)
            result= "openai error, retry"
            print("openai error, retry")
            time.sleep(2)
    return result

def llm_judge_answer(llm_output, answer, engine='gpt-3.5-turbo'):
    prompt = f"""
You are an expert in the medical domain. Given a correct answer, and the answer from medical student.
You need to judge whether the answer from medical student is correct, by comparing the answer from medical student with the correct answer.
Your response must be 'True' or 'False'.
If the answer is correct, please respond with 'True'.
If the answer is wrong, please respond with 'False'.
Correct answer:
{answer}
Answer from medical student:
{llm_output}
"""
    return run_llm(prompt, engine=engine)

def llm_judge_knowledge(llm_output, reference, engine='gpt-3.5-turbo'):
    prompt = f"""
You are an expert in the medical domain. Given a piece of medical content, and the reference information.
You need to judge whether the information in the content is consistent with the reference information.
You must carefully read the content and the reference information to make a judgment, we require a high level of trustworthiness in medical domain.
Your response must be 'True' or 'False'.
If the information in the content is consistent with the reference information, please respond with 'True'.
If the information in the content is not consistent with the reference information, please respond with 'False'.
Medical content:
{llm_output}
Reference information:
{reference}
"""
    return run_llm(prompt, engine=engine)


def get_json_from_generated_text(text):
    # extract substring between the first '{' and the last '}'
    start = text.find("{")
    end = text.rfind("}")
    json_str = text[start:end+1]
    json_obj = json.loads(json_str)
    
    return json_obj

def llm_decompose_reasoning(llm_output, engine='gpt-4o'):
    prompt = f""" You are a pattern-following humor comparator for New Yorker Caption Contest. 
Given a paragraph of reasoning justifying why one caption is funnier than other, you need to decompose the reasoning into individual steps.
Each step should contain four parts:
1. "id" : a unique number for the step
2. "planning" : a brief description of the main idea of the step. "planning" parts from all steps should form a coherent and logical sequence.
3. "knowledge" : a detailed description of the knowledge taken in this step. "knowledge" is taken based on the "planning" part of the step, and should contain specific medical knowledge or procedures.
4. "step_text" : sentence(s) from the input reasoning paragraph that corresponds to this step.

### Output Format:
Strictly follow the JSON structure below. 

```json
{{
"Steps: [
    {{"id" : 1, "planning" : "Planning for step 1", "knowledge" : "Knowledge for step 1", "step_text" : "Corresponding sentence(s) from the input reasoning paragraph"}},
    {{"id" : 2, "planning" : "Planning for step 2", "knowledge" : "Knowledge for step 2", "step_text" : "Corresponding sentence(s) from the input reasoning paragraph"}},
    ...
]
}}
```

### Input Reasoning paragraph:
{llm_output}
"""
    res_text = run_llm(prompt, engine=engine)
    return get_json_from_generated_text(res_text)

def llm_generate_query(llm_output, engine='gpt-4o'):
    prompt = f""" You are a helpful, pattern-following medical assistant.
Given a paragraph of medical content, which may contain multiple knowledge points.
You need to generate a concise query for provided content.
The query should be a concise question, which will be used to retrieve relevant information from a medical database.
If there is no specific knowledge point in the input content (such as a general introduction, summary of symptoms, etc.), you set the query as "None".

### Output Format:
If there are multiple knowledge points in the input content, you should strictly follow the JSON structure below.

```json
{{
"query" : "Query for the medical content or None",
}}
```

Otherwise there is no specific knowledge point in the input content, you set the query as "None".

### Example:

Input Content: "The patient is a 79-year-old woman on peritoneal dialysis presenting with sudden shortness of breath and a right-sided pleural effusion. Notably, she does not have fever or cough."
Output: {{"query": "None"}} (because this content are only describing the patient's symptoms, not a specific knowledge point)

Input Content: "The pleural effusion is described as clear and yellow with a low nucleated cell count, mostly lymphocytes, which suggests it is not due to an infection."
Output: {{"query": "What is the cause of pleural effusion with clear and yellow fluid and low nucleated cell count?"}} (this content describes a specific knowledge point)

### Input Knowledge point:
{llm_output}

Output:
"""
    res_text = run_llm(prompt, engine=engine)
    return get_json_from_generated_text(res_text)

def llm_check_planning_consistency(plan1: List[str], plan2: List[str], engine='gpt-4o'):
    prompt = f""" You are a helpful, pattern-following medical assistant.
Given two lists of planning lists, you need to check if the two lists are consistent with each other.
Two lists are considered consistent if they form a similar and coherent sequence of steps, and the steps in the two lists are in the same order.
Your response must be one word: 'True' or 'False'.
If the two lists are consistent, please respond with 'True'.
If the two lists are not consistent, please respond with 'False'.

Planning list 1:
{plan1}

Planning list 2:
{plan2}
"""
    return run_llm(prompt, engine=engine)

        