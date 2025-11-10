import os
import pickle
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from services import logging_config
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from datetime import datetime
from .state import AnnotatorState
from .schema import AgentDecision, TaskStatus
from .tools import update_condition_mapping, search_for_condition_mapping
from langgraph.prebuilt import ToolNode


logger = logging_config.get_logger(__name__)
load_dotenv(find_dotenv())

class AnnotatorAgent:
    def __init__(self, user_instructions:str, update: bool=False, data:pd.DataFrame = None, ):
        self.user_instructions = user_instructions
        if update:
            self.update_mode = True
        else:
            self.update_mode = False
        self.annotator = ChatOpenAI(base_url=os.getenv("LOCAL_LLM_URL"), model = os.getenv("LOCAL_LLM"))
        self.tools = [update_condition_mapping, search_for_condition_mapping]
        self.system_prompt = self._return_system_prompt()
        self.annotator_agent_state = self._get_annotator_agent_state(data)
        self.workflow_graph = self._create_agent_graph()
        
    
    #Setup state
    def _get_annotator_agent_state(self, df):
        if self.update_mode:
            state = self._recover_state()
        else:
            state = AnnotatorState(
                messages = [SystemMessage(content=self._return_system_prompt())],
                to_do_list = None,
                original_data = df,
                working_data = df,
                final_data = None,
                report = None,
                data_location = os.getenv("DATA_LOC"),
                workflow = "not_started",
                last_run=datetime.now(),
            )
        return state


    #Define Agent Nodes

    def _initiate_workflow(self, state: AnnotatorState):
        pass

    def _check_to_do_list(self, state: AnnotatorState):
        pass

    def _set_current_task(sefl, state: AnnotatorState):
        pass
    
    def _work_on_a_task(self, state: AnnotatorState):
        """
        Let the LLM work on a task by calling appropriate tools.

        Returns:
            state with LLM's response (including any tool calls)
        """

        # Create prompt asking LLM to work on the task
        task_prompt = f"""Use the appropriate tool to complete your task:
- update_condition_mapping: For mapping unmapped conditions to existing MeSH terms
- search_for_condition_mapping: For extracting conditions from trial context when information is missing/unclear

Call the tool that best fit best the task."""
        state.messages = state.messages + HumanMessage()
        # Invoke LLM with tools bound - it will decide which tool to call
        response = self.annotator.bind_tools(self.tools).invoke(
            state.messages +
            [HumanMessage(content=task_prompt)]
        )

        # Add LLM response to messages (includes tool calls if any)
        state.messages.append(response)
        logger.info(f"LLM response for task '{next_task_name}': {response.content if response.content else 'Tool call made'}")

        return state

    def _execute_tools(self, state: AnnotatorState):
        """
        Execute any tool calls made by the LLM.

        Returns:
            state with tool results added to messages
        """
        # Get the last message (should contain tool calls)
        last_message = state.messages[-1]

        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.warning("No tool calls found in last message")
            return state

        logger.info(f"Executing {len(last_message.tool_calls)} tool call(s)")

        # Create ToolNode to execute the tools
        tool_node = ToolNode(self.tools)

        # Execute tools and get results
        result_state = tool_node.invoke(state)

        logger.info("Tool execution completed")
        return result_state
    
    def _update_todo_list(self, state: AnnotatorState):
        """
        Update to-do list by marking completed tasks.

        Returns:
            state with updated to_do_list
        """
        # Find task that is IN_PROCESS and mark as COMPLETED
        pass

    #Routing Functions
    def _should_continue(self, state: AnnotatorState) -> str:
        """Use LLM to decide whether to continue working or end."""
        # Get pending tasks
        pending_tasks = [task_name for task_name, status in state.to_do_list.items() if status == TaskStatus.NOT_STARTED]
        # Create decision prompt for LLM
        decision_prompt = f"""Based on the current to-do list status, decide whether to continue working or end.

To-do list status:
- Pending tasks: {len(pending_tasks)}
- Tasks: {', '.join(pending_tasks) if pending_tasks else 'None'}

Should you:
1. "work_on_a_task" - Continue working on the next task
2. "end" - Stop working (all tasks completed)
"""
        # Get LLM decision using structured output
        response = self.annotator.with_structured_output(AgentDecision).invoke(state.messages + [HumanMessage(content=decision_prompt)])
        logger.info(f"LLM decision: {response.decision} - {response.reasoning}")
        # Return the decision directly (already in correct format)
        return response.decision if response.decision != "end" else END

    #Setup Agent Graph
    def _create_agent_graph(self):
        """
        Create the agent workflow graph.

        Graph structure:
        START → check_todo_list → [conditional routing]
                                 → work_on_a_task → execute_tools → update_todo_list → check_todo_list (loop)
                                 → END (if no tasks)
        """
        # Create graph
        workflow = StateGraph(AnnotatorState)

        # Add nodes
        workflow.add_node("work_on_a_task", self._work_on_a_task)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("update_todo_list", self._update_todo_list)

        # Add edges
        workflow.add_edge(START, "check_todo_list")
        workflow.add_edge("work_on_a_task", "execute_tools")
        workflow.add_edge("execute_tools", "update_todo_list")
        workflow.add_edge("update_todo_list", "check_todo_list")
        workflow.add_conditional_edges(
            "check_todo_list",
            self._should_continue,
            {
                "work_on_a_task": "work_on_a_task",
                END: END
            }
        )
        # Compile graph
        return workflow.compile()

    
    #General Helper Functions
    def _persist_state(self):
        state_file = "saved_states/last_run.pkl"
        try: 
            with open(state_file, 'wb') as f:
               pickle.dump(self.state, f)
        except Exception as e:
            logger.warning(f"Warning - unable to save state due to error: {e}")

    def _recover_state(self):
        state_file = "saved_state/last_run.pkl"
        if Path(state_file).exists():
            try:
                with open(state_file, "rb") as f:
                    last_state = pickle.load(f)
                if last_state:
                    logger.info(f"Successfully loaded state from last run on {self.state.last_run.strftime("%Y-%m-%d %H:%M:%S")}.")
                    return last_state
                
            except Exception as e:
                logger.error(f"Warning, unable to load last state due to error: {e}")
                return None
        else:
            logger.info(f"Warning - No prior state exists")
            return None
    
    def _return_system_prompt(self):
        return """You are an Annotator Agent specialized in mapping clinical trial condition data to standardized MeSH (Medical Subject Headings) terms.

Your role:
- Process clinical trial data to ensure all medical conditions are properly mapped to MeSH terms
- Use update_condition_mapping to map unmapped conditions using existing MeSH term knowledge
- Use search_for_condition_mapping to extract conditions from trial context when condition fields are unclear or missing
- Work through your to-do list systematically, completing each task before moving to the next
- Maintain accurate records of all mappings created and trials affected

Your tools:
1. update_condition_mapping: Maps unmapped conditions to existing MeSH terms using GPT-5 and validation
2. search_for_condition_mapping: Extracts medical conditions from trial context and finds MeSH terms using mesh_mapper service

You follow a structured workflow:
1. Check your to-do list for pending tasks
2. Execute the next pending task using the appropriate tool
3. Update your to-do list to mark the task as completed
4. Repeat until all tasks are done

Be thorough, accurate, and systematic in your work."""



