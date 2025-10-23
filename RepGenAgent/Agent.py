from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.schema import Document
from typing import List, Literal, Any
from .Output import Output

from langchain.agents import create_react_agent,AgentExecutor

from BK.db import DB
from .Source import db
from langchain_community.utilities import SQLDatabase

# Report
class GetMode(object):
    def __init__(self, llm):
        self.llm = llm
        self.prompt = self.__create_prompt()
        self.chain = self.prompt | self.llm.with_structured_output(Output.GetMode)

    def __create_prompt(self):
        template = """사용자 입력을 분석하여 mode를 task, report, ect로 구분한 후, mode를 그렇게 판단한 이유는 reason에 2~3줄로 간략하게 작성하세요.
        task:
        - 야구 관련 데이터를 DB에서 추출하는 요청
        - 야구 관련 데이터를 분석하는 요청
        - 야구 관련 데이터 시각화하는 요청

        report:
        - 야구 관련 리포트를 작성하는 요청

        ect:
        - task와 report에 속하지 않는 요청
        - 야구와 관련되지 않은 요청
        
        사용자 요청:
        {query} 
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt
        
    def invoke(self, query:str):
        answer = self.chain.invoke({'query':query})
        return answer

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'
    
class PassiveGoalCreator(object):
    def __init__(self, llm):
        self.llm = llm
        self.prompt = self.__create_prompt()
        self.chain = self.prompt | self.llm.with_structured_output(Output.PassiveGoalCreator)

    def __create_prompt(self):
        template = """너는 야구전문가야. 사용자 입력을 분석하여 명확한 목표와 그에 해당하는 문서 제목을 생성해주세요.
            요건
            1. 사용자의 입력을 바탕으로 꾸밈없이 명확한 톤으로 다음 LLM이 처리할 수 있도록 문장을 생성하시오.
            2. 기간에 대한 범위가 없다면 가장 최신 시즌을 고려하여 작성하십시오. 
            3. 사용자의 요청과 생성된 목적들을 기반으로 문서 제목을 생성해주십시오. 
            다만 기간에 대한 사용자의 정확한 요청이 없을 경우 제목을 작성할 때 기간을 명시하는 텍스트는 제외하십시오.
            사용자 입력: {query}
            """
            
            # 사용자의 부정확한 입력을 다음 LLM이 처리할 수 있도록 명확하게 한문장의로 재작성하는 작업 담당
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def invoke(self, query):
        answer = self.chain.invoke({'query':query})
        return {
            "goal": answer.description,
            "report_title": answer.report_title,
            }

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'
    

class GetReportInfo(object):
    def __init__(self, vectorstore: Chroma, llm):
        self.llm = llm
        self.prompt = self.__create_prompt()
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(d.page_content for d in docs) if docs else "(no context)"
        
        self.chain = (
        RunnablePassthrough.assign(
            # 입력 dict -> "query"만 뽑아 retriever에 넣고 -> 문자열로 포맷
            context = RunnableLambda(itemgetter("query")) | self.retriever | RunnableLambda(format_docs)
        )
        | self.prompt
        | self.llm.with_structured_output(Output.GetReportInfo)
        )
    def __create_prompt(self):
        template = """
            당신은 엄격하고 객관적인 문서 유효성 검사관입니다. 아래에 제시된 **[사용자 목표/질문]**과 **[검색된 문서]**를 철저하게 비교하여, 문서가 목표를 설명하는 데 충분한 정보를 제공하는지 여부를 판단하세요.

            [판단 기준]
            YES:
            - 문서가 **[사용자 목표/질문] 전체 주제의 본질적인 목적과 범위**를 직접적으로 다루고 있으며, 그 내용을 바탕으로 목표를 완전하게 설명하거나 보고서를 작성할 수 있을 정도로 충분한 정보를 포함한 경우.
            - 문서의 제목과 주요 내용이 질문의 중심 주제와 명확히 일치하며, 부수적인 세부 주제가 아니라 **전체 주제를 대표**하는 경우.

            NO:
            - 문서가 질문의 **주요 목차의 일부이거나 구성요소만 다루는 경우.**
            - 문서가 질문의 **핵심 키워드 일부만 포함하지만**, 그 내용을 통해 전체 목표를 충족할 수 없는 경우.
            - 문서가 질문의 주제를 **간접적으로 언급**하거나, 일부 관련된 정보만 제공하는 경우.
            - 문서가 단순히 데이터, 사례, 혹은 부분적 설명만 포함하여 **질문 전체에 대한 답변이나 초안을 구성할 수 없는 경우.**

            유의사항:
            - 핵심 키워드가 일부 등장하더라도, 그 내용이 전체 주제의 맥락과 목적을 포괄하지 않으면 반드시 NO로 판단해야 합니다.
            - 질문이 포괄적인 목적을 가진 경우, 문서가 그 **전체 목적을 충족하지 않으면 YES로 분류할 수 없습니다.**

            마지막으로 증거를 남기기위해 검색된 문서에 대해서 그대로 들고와주십시오.
            
            목표:
            {query}
            
            검색된 문서:
            {context} 
            """ 
                # 키워드 위주로 비교하며 정해진 목표가 문서로서 완성될 수 있는지 판단
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def invoke(self, query:dict):
        answer = self.chain.invoke({'query':query})
        return {
            "doc_val_response": answer.doc_val_response,
            "doc_val_reason": answer.doc_val_reason,
            "documents": answer.documents
            }
        
    def ainvoke(self, query:dict):
        answer = self.chain.ainvoke({'query':query})
        return {
            "doc_val_response": answer.doc_val_response,
            "doc_val_reason": answer.doc_val_reason,
            "documents": answer.documents
            }
    
    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'
    
class GoalOptimizer(object):
    def __init__(self, llm):
        self.llm = llm
        self.prompt = self.__create_prompt()
        self.chain = self.prompt | self.llm.with_structured_output(Output.GoalOptimizer)

    def __create_prompt(self):
        template = """당신은 야구 관련된 리포트 작성을 위한 목표 설정 전문가입니다.
        주어진 원래목표와 주어진 문서(계획서)를 기반으로 목차별로 달성 가능한 세부적인 목표를 생성하십시오.
        [원래목표]
        {query}

        [지시사항]
        1. 주어진 문서에서의 목차번호와 해당 목차의 제목을 반드시 세부적인 목표와 같이 명시해주십시오.
        2. 주어진 문서의 목차별로 최종적으로 달성해야할 목표를 아래와 같이 매핑하여 명시해주십시오.
            - 표, Table : 데이터 반환
            - Chart : 파이썬 코드 반환
            
        3. 원래 목표의 범위를 기준으로 반드시 주어진 문서의 목차별로 달성해야될 목표를 작성하십시오.
        4. 원래 목표를 그대로 반환하지 않고 아래 주어진 문서의 목차별로 달성해야될 목표를 작성하십시오.
        5. 문서의 목차 외 다른 목차를 구성하지 않습니다.
        
        [주어진 문서]
        {docs}
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def invoke(self, query, docs):
        answer = self.chain.invoke({'query':query, 'docs':docs})
        return answer.text

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'
    
class TaskDecomposer(object):
    def __init__(self, llm):
        self.llm = llm
        self.prompt = self.__create_prompt()
        self.chain = self.prompt | self.llm.with_structured_output(Output.TaskDecomposer)

    def __create_prompt(self):
        template = """태스크: 주어진 목표를 콘텐츠만을 기준으로 실행 가능한 태스크로 분해해 주세요.
        요건
        1. 다음 행동만으로 목표를 달성할 것. 절대 지정된 이외의 행동을 취하지 말 것.
        - 사용자의 요청에 따라 적절한 데이터를 불러와 데이터프레임으로 변환합니다.
        - 사용자 요청에 따라 DB에서 적절한 데이터를 불러와서 시각화할 수 있는 코드를 작성합니다.
        - DB외에 추가로 필요한 정보가 필요할 경우만 판단해서 인터넷 검색을 통해서 정보를 추가합니다.
        2. 각 태스크는 구체적으로 상세하게 기재하며, 단독으로 실행 및 검증 가능한 정보를 포함할 것. 추상적인 표현을 일절 포함하지 말것
        3. 목표에 작성된 콘텐츠 수를 기준으로 태스크를 최소한으로 생성할 것.
        4. 아래와 같이 구성할 것
        - contents_id: 콘텐츠 번호. 기존 콘텐츠 명 , type: 결과유형 [Chart, Table], contents: 태스크 내용
        5. 태스크는 실행 가능한 순서로 리스트화 할 것
        목표: {query}
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def invoke(self, query):
        answer = self.chain.invoke({'query':query})
        return answer.tasks

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'
    
class ExecuteTask(object):
    def __init__(self, llm, tools:list):
        #self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0, top_p=1)
        self.llm = llm
        self.tools = tools
        self.prompt = self.__create_prompt()
        self.agent = self.__create_agent()

    def __create_prompt(self):
    
        template = """다음 태스크를 실행하고 상세한 답변을 제공해주세요. 당신은 다음 도구에 접근할 수 있습니다:
    
        {tools}
        
        다음 형식을 사용하세요:
        
        Question: 답변해야 하는 입력 질문
        Thought: 무엇을 할지 항상 생각하세요.
        Action: 취해야 할 행동, [{tool_names}] 중 하나여야 합니다.
        Action Input: 행동에 대한 입력값
        Observation: 행동의 결과... (이 Thought/Action/Action Input/Observation의 과정이 N번 반복될 수 있습니다.)
        Thought: 이제 최종 답변을 알겠습니다.
        Final Answer: 원래 입력된 질문에 대한 최종 답변
        
        ## 추가적인 주의사항
        - 반드시 [Thought/Action/Action Input format] 이 사이클의 순서를 준수하십시오. 항상 Action 전에는 Thought가 먼저 나와야 합니다.
        - 한 번의 검색으로 해결되지 않을 것 같다면 문제를 분할하여 푸는 것이 중요합니다.
        - Action Input은 정확하게 필요한 요소들로 생성합니다. query: 등과 같은 접두어 사용금지
        - 정보가 취합되었다면 불필요하게 사이클을 반복하지 마십시오.
        - 묻지 않은 정보를 찾으려고 도구를 사용하지 마십시오.
        - 가능한 구체적인 사실이나 데이터를 제공하세요.
        - 차트를 생성하는 코드를 작성하는 요청이 들어오면, 차트를 생성하는 순수한 파이썬 코드만 답변하세요. 
        - 데이터프레임을 반환하는 경우는 마크다운 형식말고 순수한 dictionary 형태(key:[value])로 작성하고, 차트는 파이썬 코드는 마크다운 형식 말고 순수한 파이썬 코드만 생성
        
        시작하세요!
        Question: {task}
        Thought: {agent_scratchpad}
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def __create_agent(self):
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        return executor
        
    def invoke(self, task):
        answer = self.agent.invoke({'task':task})
        return answer['output']

    def ainvoke(self, task):
        answer = self.agent.ainvoke({'task':task})
        return answer['output']

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'
    
class ResultAggregator(object):
    def __init__(self, llm, db:SQLDatabase=db):
        self.llm = llm
        self.db = db
        self.tb_desc = self._get_db_desc()
        self.select_tb_chain = self.__create_prompt('select_table') |  self.llm.with_structured_output(Output.DataLoader)
        self.chain = self.__create_prompt('write_summary') | self.llm | StrOutputParser()

    def __create_prompt(self, prompt_type:str):
 
        prompt = {
            'select_table':
                  """당신은 SQL 쿼리 전문가입니다. 사용자의 요청을 SQL 쿼리로 변환하는 임무를 맡습니다. 
                  아래 사용자 요청을 이해한 후 테이블 정보를 참고하여 필요한 테이블명을 모두 선정한 후 리스트 형태로 반환하시오.
                  
                  사용자 요청
                  {query}
                  
                  테이블 정보
                  {table_desc}
                  
                  예시
                  - 사용자 요청: 팀단위 전력분석 보고서 작성에 대한 상세 정보
                  - 답변: ['tb_st_team_standings_stats']""",
            
            
            'write_summary': """주어진 목표:
            {optimized_goal}
            
            조사결과:
            {results}
            
            너는 조사결과와 주어진 목표를 바탕으로 보고서 결론을 작성하는 야구 전문가야. 다음과 같은 지시 사항을 준수하여 응답을 생성해줘
            [지시사항]
            - 조사결과를 활용하여 주어진 목표를 달성할 수 있는 글을 600자 내외로 작성해주십시오.
            - 타이틀과 같은 마크다운요소는 제외하고 줄글로 작성하되 중요하다고 생각하는 단어말고 문장을 bold처리를 해줘서 해당 보고서의 결론을 작성하여주십시오.
            - 컬럼 메타 정보를 반드시 참조하여 결과를 해석해주십시오.
            
            컬럼 메타 정보:
            {column_meta}
            """
            }
    
        prompt = ChatPromptTemplate.from_template(prompt[prompt_type])

        return prompt
        
    def invoke(self, optimized_goal:str, results:str):
        target_tables = self.select_tb_chain.invoke({'query':optimized_goal, 'table_desc':self.tb_desc}).target_tables
        table_info = self.db.get_table_info(table_names=target_tables, get_col_comments=True)
        column_info = table_info.split('*\n')[1] 
        answer = self.chain.invoke({'optimized_goal':optimized_goal, 'results':results, 'column_meta':column_info})
        return answer
    
    def _execute_query(self, query):
        rdb = DB('pg', 'postgres')
        data = rdb.read_table(query)
        return data # 여기에 테이블정보를 추가하는 방법
    
    def _get_db_desc(self):
        query = """SELECT ps.relname as table_nm, pd.description AS table_desc
                   FROM pg_stat_user_tables ps
                   JOIN pg_description pd
                   ON ps.relid = pd.objoid AND pd.objsubid = 0
                   where ps.schemaname = 'doosan' """
        return {k:v for k, v in self._execute_query(query).values}

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'

# Task
class SingleGettInfo(object):
    def __init__(self, llm):
        self.llm = llm
        self.prompt = self.__create_prompt()
        
        self.chain = (
        RunnablePassthrough.assign(query=itemgetter("query"))
        | self.prompt
        | self.llm.with_structured_output(Output.GetReportInfo)
        )

    def __create_prompt(self):
        template = """
        너는 "요구사항 명확성 심사관"이다.  
        사용자의 질문이 주어진 목표를 실행하기에 충분히 구체적이고 실행 가능한지 유형별로 판단해야 한다.  

        판단 규칙:
        1. 질문 유형별로 아래 기준을 따른다.
        [1] 시각화
        - 조회한 데이터가 막대그래프, 선그래프, 산점도, 히스토그램, 박스플롯, 파이차트 6가지 유형 각화 요청일 경우:
        - 데이터셋 또는 시각화에 필요한 컬럼이 명시되어 있는가?
        - 조회한 데이터가 차트를 그리기에 충분한 조건 및 비교 대상(예: 팀, 시즌, 선수 등)이 포함되어 있는가?중 하나로 선택해서 표현할 수 있는가?

        [2] 데이터 추출 요청일 경우:
        - 어떤 데이터를 어떤 조건으로 조회할지 명시되어 있는가?
        - 필요한 컬럼 또는 기준(예: 팀명, 연도, 통계 항목 등)이 포함되어 있는가?

        [3] 분석 요청일 경우:
        - 분석의 목적(예: 상관관계, 추세, 비교 등)이 명확한가?
        - 분석 대상 데이터와 기준(예: 팀, 시즌, 변수 등)이 구체적인가?
        
        2. 위 조건이 충족되어 있으면 "YES", 불충분하거나 모호하면 "NO"로 판단한다.
        
        3. 판단 이유를 한 줄 요약한다.

        JSON 출력 형식:
        {{
        "doc_val_response": "YES" | "NO",
        "doc_val_reason": "한 줄 요약"
        }}

        사용자 질문:
        {query}
        """
                
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def invoke(self, query:dict):
        answer = self.chain.invoke({'query':query})
        return {
            "doc_val_response": answer.doc_val_response,
            "doc_val_reason": answer.doc_val_reason
            }
        
    def ainvoke(self, query:dict):
        answer = self.chain.ainvoke({'query':query})
        return {
            "doc_val_response": answer.doc_val_response,
            "doc_val_reason": answer.doc_val_reason
            }
    
    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'

class SingleResultInspect(object):
    def __init__(self, llm):
        self.llm = llm
        self.prompt = self.__create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser() | RunnableLambda(self.__clean_answer)

    def __create_prompt(self):
        template = """
        주어진 목표:
        {goal}
        
        조사결과:
        {results}
        
        주어진 목표에 대해서 리스트안의 조사 결과를 활용하여 다음 지시에 기반한 응답을 무조건 하나만 생성해 주세요.
        - 주어진 목표에 따라 순수한 데이터, 코드만 정제해서 출력한다.
        - 리스트 안의 조사결과가 하나일 경우 리스트가 아닌 text형식으로 조사결과를 ouput으로 지정한다. (수행 과정, 로그, 코멘트 등 작성 금지)
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def __clean_answer(self, answer:str):
        answer = answer.replace('```', '')
        answer = answer.replace('python', '')
        answer = answer.replace('plt.show()', '')
        return answer
    
    def invoke(self, goal:str, results:str):
        answer = self.chain.invoke({'goal':goal, 'results':results})
        return answer

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'

# Response
class ResponseCustomer(object):
    def __init__(self, llm):
        self.llm = llm
        self.prompt = self.__create_prompt()
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        
    def __create_prompt(self):
        template = """
                당신은 챗봇입니다.
                아래 '결정 사유'를 참고해, 사용자가 무엇을 원했는지 확인하고 아래의 조건을 선택해서 답변을 생성하십시오.
                이전 질문이 있다면 이것도 기억하고 답변을 생성하십시오.
                
                조건 1: 결정 사유가 라우팅 실패에 관련된 내용일 경우
                - 형식 가이드(3~5문장, 한국어, 공손하지만 간결하게):
                1) 확인: 사용자의 의도를 한 문장으로 재진술 (예: "~정보를 요청하셨군요.")
                2) 한계: 현재 참조 가능한 자료에 무엇이 없는지 명확히 (예: "현재 제가 참고할 수 있는 정보에는 ~가 포함되어 있지 않습니다.")
                3) 대안: a) 가능한 주제로 이어가기

                조건 2: 결정 사유가 모드에 대한 선택에 대한 이유일 경우
                - 형식 가이드(3~5문장, 한국어, 공손하게 간결하게):
                1) 확인: 사용자의 의도를 한 문장으로 재진술 (예: "~정보를 요청하셨군요.")
                2) 답변: 사용자의 의도에 대해서 답변을 한다.
                3) 한계: 최대한 꾸며내거나 부풀리지 않고 정확하게 답변하려고 노력하여야한다. 그리고 답변이 정확하지 않을 수 있다는 사실을 분명히 한다.

                [사용자 질문]
                {question}

                [결정 사유]
                {doc_val_reason}
                
                [이전 질문]
                {prev_user_question}
                """ 
        prompt = ChatPromptTemplate.from_template(template)
        return prompt
    
    def invoke(self, query:dict, doc_val_reason:str, prev_user_query:str):
        return self.chain.invoke({'question':query,
                                  'doc_val_reason': doc_val_reason,
                                  'prev_user_question': prev_user_query})

    def ainoke(self, query:dict, doc_val_reason:str, prev_user_query:str):
        return self.chain.ainvoke({'question':query,
                                  'doc_val_reason': doc_val_reason,
                                  'prev_user_question': prev_user_query})
        
    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'
