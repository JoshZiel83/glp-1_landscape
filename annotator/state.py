from dataclasses import dataclass
from langgraph.graph import MessagesState
import pandas as pd
from datetime import datetime
from annotator.schema import TaskStatus

 
@dataclass
class AnnotatorState:
    messages: MessagesState
    to_do_list: dict | None
    original_data: pd.DataFrame 
    working_data: pd.DataFrame | None
    final_data: pd.DataFrame | None
    report:str | None
    data_location: str 
    workflow: TaskStatus
    last_run: datetime 