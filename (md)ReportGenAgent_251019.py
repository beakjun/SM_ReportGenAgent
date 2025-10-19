# DataLoader
class DataLoader(object):
    def __init__(self, db:SQLDatabase = db):
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
        self.db = db
        self.tb_desc = self._get_db_desc()
        self.write_query = create_sql_query_chain(llm, db, prompt=self.__create_prompt('write_query'))
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
                  - 모든 컬럼명에는 ""로 감싸주며, as를 통해서 alias 하더라도 ""를 사용할 것.
                  - Select외 나머지 DDL 사용 금지.
                  - 최종 출력은 SQL 쿼리만 출력.
                  - 어떤 경우에도 "SQLQuery:", "Answer:", "Output:" 등의 접두어를 포함하지 않는다.
                  - 반드시 SQL문만 순수하게 출력한다. (SELECT로 시작해야 함)
                  - 지표별 리그 내 순위 생성이 명시된 콘텐츠에서만 순위 생성 시 반드시 아래와 같은 규칙을 따라줘:
                    * 논리적 순서: 먼저 해당 시즌 데이터만 필터링하고, 그 안에서 모든 팀의 각 지표에 대한 순위를 계산해줘. 마지막으로 내가 원하는 특정 팀의 데이터만 필터링해서 보여줘.
                    * 포함할 컬럼: "팀 명", "시즌"을 제외한 지표들의 값.
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
        rdb = DB('pg', 'postgres')
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

class TaskInfo(BaseModel):
    contents_id: str = Field(..., description="콘텐츠 목차 + 콘텐츠 명")
    type: Literal["Chart", "Table"] = Field(..., description="콘텐츠 결과 유형")
    contents: str = Field(..., description="태스크 설명")

class Output(object):
    class GetMode(BaseModel):
        mode: Literal['task', 'report', 'ect'] = Field(..., description='사용자 요청을 분석하여 요청 유형을 task, report, ect로 분류')
        reason: str = Field(default='', description='사용자 요청을 분류한 근거를 2~3로 작성하세요.')
        query_id: str = Field(default='', description='Python에서 UUID로 생성되는 고유 ID')
        
    class PassiveGoalCreator(BaseModel):
        report_title: str = Field(..., description='보고서 제목')
        description: str = Field(..., description='목표 설명')
        
        
        @property
        def text(self) -> str:
            return f'{self.description}'
            
    class GoalOptimizer(BaseModel):
        description:str = Field(..., description='목표 설명')
    
        @property
        def text(self) -> str:
            return f'{self.description}'

    
    class TaskDecomposer(BaseModel):
        tasks: list[TaskInfo] = Field(
            default_factory=list,
            min_item=3,
            max_item=8,
            description='3~8개로 분해된 테스크')
    
    class GetReportInfo(BaseModel):
        doc_val_response: Literal["YES","NO"] = Field(default='YES', description='문서 검증 결과')
        doc_val_reason:str= Field(default='', description='문서 검증 증거')
        documents:str = Field(default='', description='검증에 참조한 문서')

    class DataLoader(BaseModel):
        target_tables: list[str] = Field(description='사용자 요청을 분석하여 필요한 테이블명을 저장')