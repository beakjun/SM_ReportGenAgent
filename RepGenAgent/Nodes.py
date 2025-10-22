from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal, Any
from functools import wraps
from typing import Annotated
from datetime import datetime 
from operator import add
from langgraph.graph import START, END, StateGraph

from .Agent import *
from .Tools import report_tools, task_tools
from .Source import agent_llm, chat_llm, vectorstore

import uuid

class Turn(BaseModel):
    user_query: str=Field(default_factory=list, description='유저 쿼리')      
    mode: Literal['task', 'report', 'ect'] = Field(default='', description="모드")
    ai_answer: str=Field(default_factory=list, description='답변')                           
    ts: datetime=Field(default_factory=datetime.now())

class State(BaseModel):
    query: str = Field(..., description='사용자가 입력한 쿼리')
    query_id: str = Field(default='', description='사용자가 입력한 쿼리 고유 ID')
    history: Annotated[list[Turn], add] = Field(default_factory= list, description=' 이전 기록')
    goal: str = Field(default='', description='사용자가 입력한 쿼리에서 목표 추출')
    report_title: str = Field(default='', description='사용자요청을 기반으로 생성된 보고서 제목')
    mode: Literal['task', 'report', 'ect'] = Field(default='', description='사용자 요청 유형')
    doc_val_response: Literal["YES","NO"] = Field(default='YES', description='문서 검증 결과')
    doc_val_reason:str= Field(default='', description='문서 검증 증거')
    documents:str = Field(default='', description='검증에 참조한 문서')
    optimized_goal: str = Field(default='', description='최적화된 목표') 
    tasks: list = Field(default_factory=list, description='실행할 테스크 리스트')
    current_task_index: int = Field(default=0, description='현재 실행 중인 테스크 변호')
    results: list = Field(default_factory=list, description='실행 완료된 테스크 결과 리스트')
    final_output: str = Field(default='', description='최종 출력 결과')
    save_dir: str = Field(default='', description='프롬프트id별 저장 경로')

def node_logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{func.__name__} 시작")
        result = func(*args, **kwargs)
        print(f"{func.__name__} 완료")
        return result
    return wrapper

@node_logging
def get_mode(state:State):
    state = state.dict()
    query = state['query']
    answer = GetMode(llm=agent_llm).invoke(query)
    answer.query_id = uuid.uuid1().hex
    return {
            'mode':answer.mode,
            'doc_val_reason':answer.reason,
            'query_id': answer.query_id,
            'tasks': [],
            'results': [],
            'current_task_index': 0
            }

# 리포트
@node_logging
def get_goal(state:State):
    state = state.dict()
    query = state['query']
    mode = state['mode']
    query_id = state['query_id']
    answer = PassiveGoalCreator(llm=agent_llm).invoke(query)
    return {'goal':answer['goal'],
            'report_title': answer['report_title'],
            'save_dir': f'./reports/{mode}/{query_id}.docs'
            }

@node_logging
def get_report_info(state:State):
    state = state.dict()
    query = state['goal']
    answer = GetReportInfo(vectorstore=vectorstore, llm=agent_llm).invoke(query)
    return {'doc_val_response': answer['doc_val_response'],
            "doc_val_reason": answer['doc_val_reason'],
            "documents": answer['documents']}

@node_logging
def optimize_goal(state:State):
    state = state.dict()
    query = state['goal']
    docs = state['documents']
    answer = GoalOptimizer(llm=agent_llm).invoke(query, docs)
    
    # results, tasks, index 초기화

    return {'optimized_goal':answer}
 
@node_logging
def decompose_tasks(state:State):
    state = state.dict()
    query = state['optimized_goal']
    answer = TaskDecomposer(llm=agent_llm).invoke(query)
    return {'tasks': answer}

@node_logging
def execute_task(state:State):
    state = state.dict()
    complete_tasks = state['results']
    current_task_index = state['current_task_index']
    task = state['tasks'][current_task_index]
    answer = ExecuteTask(tools=report_tools).invoke(task['contents'])
    task['results'] = answer
    current_task_index += 1
    return {'results': complete_tasks + [task], 'current_task_index':current_task_index}

@node_logging
def aggregate_result(state:State):
    state = state.dict()
    results = state['results']
    optimized_goal = state['optimized_goal']
    #optimized_response = state['optimized_response']
    answer = ResultAggregator().invoke(optimized_goal, results)
    
    history = {
        "user_query": state['query'],
        "ai_answer": answer,
        "ts" : datetime.now()
    }
    
    return {'final_output': answer, "history" : [history]}


# 단일 태스크
@node_logging
def single_get_info(state:State):
    state = state.dict()
    goal = state['goal']
    mode = state['mode']
    query_id = state['query_id']
    answer = SingleGettInfo(llm=agent_llm).invoke(goal)
    return {'doc_val_response': answer['doc_val_response'],
            'doc_val_reason': answer['doc_val_reason'],
            'save_dir': f'./images/{mode}/{query_id}.png'}


@node_logging
def single_execute_task(state:State):
    state = state.dict()
    goal = state['goal']
    answer = ExecuteTask(tools= task_tools).invoke(goal)
    return {'results': [answer]}



@node_logging
def single_inspect_result(state:State):
    state = state.dict()
    results = state['results']
    goal = state['goal']
    answer = SingleResultInspect(llm=agent_llm).invoke(goal, results)
    
    history = {
        "user_query": state['query'],
        "ai_answer": answer,
        "ts" : datetime.now()
    }
    
    return {'final_output': answer, 'history': [history]}

# 챗봇
@node_logging
def response_customer(state:State):
    state = state.dict()
    query = state['query']
    doc_val_reason = state['doc_val_reason']
    
    history = state.get('history', [])
    if history:
        prev_user_query = history[-1]['user_query']
    else: 
        prev_user_query="이전 질문 없음"
        
    answer = ResponseCustomer(llm=chat_llm).invoke(query = query, doc_val_reason = doc_val_reason, prev_user_query=prev_user_query)
    
    history = {
        "user_query": state['query'],
        "ai_answer": answer,
        "ts" : datetime.now()
    }
    
    return {'final_output': answer, "history" : [history]}