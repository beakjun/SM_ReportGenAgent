from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from output import Output
from source import db
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilyAnswer, TavilySearchResults

import re
from BK.db import DB

class DataLoader(object):
    def __init__(self, db:SQLDatabase):
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0, top_p=1)
        self.db = db
        self.tb_desc = self._get_db_desc()
        self.write_query = create_sql_query_chain(self.llm, db, prompt=self.__create_prompt('write_query'))
        self.select_tb_chain = self.__create_prompt('select_table') | self.llm.with_structured_output(Output.DataLoader)
        self.chain = self.write_query | RunnableLambda(self.__clean_answer) |RunnableLambda(self._execute_query)
    
    def __create_prompt(self, prompt_type:str):
        prompt = {'select_table':
                  """당신은 SQL 쿼리 전문가입니다. 사용자의 요청을 SQL 쿼리로 변환하는 임무를 맡습니다. 
                  아래 사용자 요청을 이해한 후 테이블 정보를 참고하여 필요한 테이블명을 모두 선정한 후 리스트 형태로 반환하시오.
                  
                  사용자 요청
                  {query}
                  
                  테이블 정보
                  {table_desc}
                  
                  예시
                  - 사용자 요청: 2024년 시즌에 ARI팀은 몇 승 몇 패 했어?
                  - 답변: ['tb_st_team_standings_stats']""",
                  'write_query':
                  """당신은 SQL 쿼리 전문가입니다. 사용자의 요청을 SQL 쿼리로 변환하는 임무를 맡습니다. 
                  주어진 input을 활용하여 툴을 호출한 후 나오는 DB정보를 활용해 SELECT문을 써서 {dialect}로 정의하시오.
                  
                  쿼리 작성 규칙:
                  - 주어진 정보를 깊게 생각하여 쿼리를 생성하여라.
                  - tool을 있는 그대로(축약금지, 꾸며내기 금지) 활용해 참조한 DB정보를 바탕으로 불러온 DB정보안에 들어있는 컬럼명을 반드시 참조해서 SQL문 작성.
                  - 모든 컬럼명에는 ""로 감싸주며, as를 통해서 alias 하더라도 ""를 사용할 것 그리고 기존 영문 컬럼명으로 통일할 것.
                  - 팀,시즌과 같이 조회 조건으로 활용한 컬럼명들은 쿼리 작성 시 Select문에서 반드시 제외할 것
                  - Select외 나머지 DDL 사용 금지.
                  - 최종 출력은 SQL 쿼리만 출력.
                  - 어떤 경우에도 "SQLQuery:", "Answer:", "Output:" 등의 접두어를 포함하지 않는다.
                  - 반드시 SQL문만 순수하게 출력한다. (SELECT로 시작해야 함)
                  - tool을 호출하지 않고는 절대 판단하지 말 것.
                  - Schema정보를 반드시 참조하여 테이블명을 완성시킬 것.
                  - 소수점이 긴 경우, numeric 타입으로 변환 후 둘째 자리까지 반올림한다.
                  - 어떤 경우에도 쿼리는 ```을 사용하지 않고 작성한다.
                  - LIMIT에 대한 요청이 없는 경우 전체 데이터 조회
                  - UNION ALL을 사용하여 데이터를 합쳐야할 경우 limit 구문을 사용하지 말것
                  
                  사용자 요청:
                  {input}
                  
                  테이블 정보: 
                  {table_info}
                  
                  최대 데이터 추출 수: {top_k}
                  """}
        return ChatPromptTemplate.from_template(prompt[prompt_type])

    def _execute_query(self, query):
        rdb = DB('exem', 'postgres')
        data = rdb.read_table(query)
        return data

    def _get_db_desc(self):
        query = """SELECT ps.relname as table_nm, pd.description AS table_desc
                   FROM pg_stat_user_tables ps
                   JOIN pg_description pd
                   ON ps.relid = pd.objoid AND pd.objsubid = 0
                   where ps.schemaname = 'doosan' """
        return {k:v for k, v in self._execute_query(query).values}

    def invoke(self, query:dict):
        target_tables = self.select_tb_chain.invoke({'query':query, 'table_desc':self.tb_desc}).target_tables
        table_info = self.db.get_table_info(table_names=target_tables, get_col_comments=True)
        return self.chain.invoke({'question':query, 'table_info':table_info})

    def ainoke(self, query:dict):
        target_tables = self.select_tb_chain.ainvoke({'query':query, 'table_desc':self.tb_desc}).target_tables
        table_info = self.db.get_table_info(table_names=target_tables, get_col_comments=True)
        return self.chain.ainvoke({'question':query, 'table_info':table_info})
    
    def __clean_answer(self, query):
        clean_query = re.sub('SQLQuery: {0,1}', '', query)
        clean_query = re.sub('^sql|sql', '', clean_query)
        clean_query = re.sub('```', "'''", clean_query)
        return clean_query.strip()
        
    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'

class DataAnalyst(object):
    def __init__(self, db:SQLDatabase):
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
        self.prompt = self.__create_prompt()
        self.data_loader_chain = DataLoader(db).chain
        self.chain = RunnablePassthrough.assign(data= self.data_loader_chain) | self.prompt | self.llm | StrOutputParser()

    def __create_prompt(self):
        template = """
                    당신은 데이터 분석 전문가입니다. 사용자에게 요청에 따라 데이터를 해석하고 분석하는 임무를 맡습니다. 
                
                    데이터 해석 규칙:
                    - 사용자 요청과 주어진 데이터만 활용해서 해석을 합니다.
                    - 다만 주어진 데이터로 새로운 변수를 만들어 해석할 여지가 있으면 새로운 변수를 만들어서 해석을 해야합니다.
                    - 데이터는 사용자 요청과 관련있는 데이터입니다.
                    
                    사용자 요청: 
                    {question}
                    
                    데이터:
                    {data}
                """ 
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def invoke(self, query:dict):
        return self.chain.invoke({'question':query})

    def ainoke(self, query:dict):
        return self.chain.ainvoke({'question':query})
    
    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'

class BI(object):
    def __init__(self, db:SQLDatabase):
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
        self.prompt = self.__create_prompt()
        self.data_loader_chain = DataLoader(db).chain
        self.chain = RunnablePassthrough.assign(data= self.data_loader_chain) | self.prompt | self.llm | StrOutputParser() | RunnableLambda(self.__clean_answer)

    def __create_prompt(self):
        template = """
        당신은 Python Matplotlib 및 Seaborn 시각화 전문가입니다.
        당신의 임무는 제공된 데이터(DataFrame)와 사용자 요청을 바탕으로 가장 적합한 시각화 유형을 자동으로 선택하고, 실행 가능한 코드를 생성하는 것입니다.
        제공된 데이터(DataFrame)를 분석하고, 컬럼의 데이터 타입(num / category / datetime)을 직접 정의해서 가장 적합한 시각화 유형을 자동으로 선택해야 합니다.

        1. 데이터 확인 원칙
        - 컬럼 데이터 타입 판별: num, category, datetime
        - 고유값 개수(unique count) 및 결측치 비율 확인
        - 단일 핵심 지표 여부 확인
        - 누적값 여부 확인 (시계열/누적 구조 판단)
        - 데이터를 정의할 경우 반드시 DB에서 조회된 데이터만 사용한다. (임의의 값 생성 금지)
        - 모든 컬럼 길이가 동일하도록 반드시 확인

        2. 시각화 선택 원칙
        - 데이터의 주요 컬럼 타입(num / category / datetime)을 파악한 후 가장 기본적인 그래프를 선택합니다.
        - 고급형(복합, 분포, 상관) 그래프는 사용하지 않습니다.
        - 오직 다음 기본 유형만 고려합니다:
         * 막대그래프 (Bar) : category → num 비교
         * 선그래프 (Line) : datetime → num 시계열
         * 산점도 (Scatter) : num ↔ num 관계
         * 히스토그램 (Histogram) : 단일 num 분포
         * 박스플롯 (Boxplot) : category → num 분포 비교
         * 파이차트 (Pie) : 비율 강조 필요 시

        3. 코드 생성 규칙
        - 반드시 실행 가능한 Python 코드만 작성
        - 한글이 깨지지 않도록 Python 코드 작성
        - 코드는 절대 마크다운(`````, ''') 블록 안에 넣지 말고, 실행 가능한 Python 코드만 출력하라.
        - 주석(#)은 허용하되, 불필요한 설명, 문자열, 마크다운(`````, `'''`)은 포함하지 않음
        - exec() 함수로 바로 실행 가능한 형태여야 함
        - 폰트는 라이브러리에 내장된 기본 폰트를 반드시 사용한다. (Windows 환경)
        - plt.show()는 생략
        - print()는 사용하지 않음
        - 그래프 제목, X/Y축 라벨, 주요 범례 및 색상 강조 필수
        - 시각화에 사용된 컬럼명을 코드 주석으로 명시
        - 버전과 상관없는 오류가 가장 적은 기본적인 코드로만 구성
        - 결과를 저장할 필요 없음 (코드만 생성)
        - 결과가 글자 겹침, 범례 와 차트간의 간격 등을 잘 고려해서 최대한 보기 좋을 수 있도록 Python 코드 작성

        입력 데이터:
        {data}

        사용자 요청:
        {question}
        """ 
        
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def __clean_answer(self, answer:str):
        answer = answer.replace('```', "'''")
        answer = answer.replace('python', '')
        return answer

    def invoke(self, query:dict):
        return self.chain.invoke({'question':query})

    def ainoke(self, query:dict):
        return self.chain.ainvoke({'question':query})

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}'
    
@tool(name_or_callable="DataLoaderTool", description='사용자 요청에 따라 DB에서 적절한 데이터를 불러와서 데이터 프레임으로 반환합니다.' ,return_direct=False)
def load_data(query:str) -> str:
    loader = DataLoader(db)
    return loader.invoke(query)

@tool(name_or_callable="DataAnalystTool", description='사용자 요청에 따라 DB에서 적절한 데이터를 불러온 후 분석하고 해석합니다.' ,return_direct=False)
def analysis_data(query:str) -> str:
    analyst = DataAnalyst(db)
    return analyst.invoke(query)

@tool(name_or_callable="BITool", description='사용자 요청에 따라 DB에서 적절한 데이터를 불러와서 시각화할 수 있는 코드를 작성합니다.' ,return_direct=False)
def visualize_data(question: str) -> str:
    bi = BI(db)
    return bi.invoke(question)

# tools
report_tools = [TavilySearchResults(max_results=3), load_data, visualize_data]
task_tools = [TavilySearchResults(max_results=3), load_data, visualize_data, analysis_data]