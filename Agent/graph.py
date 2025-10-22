from nodes import *
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(State)

# 노드 추가
workflow.add_node('get_mode', get_mode)

# 리포트
workflow.add_node('get_goal', get_goal)
workflow.add_node('get_report_info', get_report_info)
workflow.add_node('optimize_goal', optimize_goal)
workflow.add_node('decompose_tasks', decompose_tasks)
workflow.add_node('report_execute_task', execute_task)
workflow.add_node('aggregate_result', aggregate_result)

# 단일 태스크
workflow.add_node('single_get_goal', get_goal)
workflow.add_node('single_get_info', single_get_info)
workflow.add_node('single_execute_task', single_execute_task)
workflow.add_node('single_inspect_result', single_inspect_result)

# 챗봇
workflow.add_node('response_customer', response_customer)

# 엣지 정의
workflow.add_edge(START, 'get_mode')
workflow.add_conditional_edges('get_mode', lambda state: state.dict()['mode'], {'task':'single_get_goal', 'report':'get_goal', 'ect':'response_customer'})
workflow.add_edge('single_get_goal', 'single_get_info')
workflow.add_conditional_edges('single_get_info', lambda state: state.dict()['doc_val_response'], {'YES':'single_execute_task', 'NO': 'response_customer'})
workflow.add_edge('single_execute_task', 'single_inspect_result')

workflow.add_edge('response_customer', END)

workflow.add_edge('get_goal', 'get_report_info')
workflow.add_conditional_edges('get_report_info', lambda state: state.dict()['doc_val_response'], {'YES':'optimize_goal', 'NO':'response_customer'})
workflow.add_edge('optimize_goal', 'decompose_tasks')
workflow.add_edge('decompose_tasks', 'report_execute_task')
workflow.add_conditional_edges('report_execute_task', lambda state: state.dict()['current_task_index'] < len(state.dict()['tasks']), {True:'report_execute_task', False:'aggregate_result'})
workflow.add_edge('aggregate_result', END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

