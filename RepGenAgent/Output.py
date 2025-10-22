from pydantic import BaseModel, Field
from typing import Literal, Optional, Any

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
            min_items=3,
            max_items=8,
            description='3~8개로 분해된 테스크')
    
    class GetReportInfo(BaseModel):
        doc_val_response: Literal["YES","NO"] = Field(default='YES', description='문서 검증 결과')
        doc_val_reason:str= Field(default='', description='문서 검증 증거')
        documents:str = Field(default='', description='검증에 참조한 문서')
    
    class DataLoader(BaseModel):
        target_tables: list[str] = Field(description='사용자 요청을 분석하여 필요한 테이블명을 저장')
        
    class ResultAggregator(BaseModel):
        final_output: str = Field(default='', max_length=700 ,description='최종 출력 결과')
        
        @property
        def text(self) -> str:
            return f'{self.final_output}'