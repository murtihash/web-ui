import json
import logging
import pdb
import traceback
from typing import Optional, Type, List, Dict, Any, Callable
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import io
import platform
from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.service import Agent
from browser_use.agent.views import (
    ActionResult,
    ActionModel,
    AgentHistoryList,
    AgentOutput,
    AgentHistory,
)
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserStateHistory
from browser_use.controller.service import Controller
from browser_use.telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepTelemetryEvent,
)
from browser_use.utils import time_execution_async
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)
from browser_use.agent.prompts import PlannerPrompt

from json_repair import repair_json
from src.utils.agent_state import AgentState

from .custom_message_manager import CustomMessageManager
from .custom_views import CustomAgentOutput, CustomAgentStepInfo
from .custom_prompts import HashDealsSystemPrompt, HashDealsAgentMessagePrompt # Import new prompts

logger = logging.getLogger(__name__)


class CustomAgent(Agent):
    def __init__(
            self,
            task: str,
            llm: BaseChatModel,
            add_infos: str = "",
            browser: Browser | None = None,
            browser_context: BrowserContext | None = None,
            controller: Controller = Controller(),
            use_vision: bool = True,
            use_vision_for_planner: bool = False,
            save_conversation_path: Optional[str] = None,
            save_conversation_path_encoding: Optional[str] = 'utf-8',
            max_failures: int = 3,
            retry_delay: int = 10,
            system_prompt_class: Type[SystemPrompt] = SystemPrompt, # Default to general SystemPrompt
            agent_prompt_class: Type[AgentMessagePrompt] = AgentMessagePrompt, # Default to general AgentMessagePrompt
            max_input_tokens: int = 128000,
            validate_output: bool = False,
            message_context: Optional[str] = None,
            generate_gif: bool | str = True,
            sensitive_data: Optional[Dict[str, str]] = None,
            available_file_paths: Optional[list[str]] = None,
            include_attributes: list[str] = [
                'title',
                'type',
                'name',
                'role',
                'tabindex',
                'aria-label',
                'placeholder',
                'value',
                'alt',
                'aria-expanded',
            ],
            max_error_length: int = 400,
            max_actions_per_step: int = 10,
            tool_call_in_content: bool = True,
            initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
            # Cloud Callbacks
            register_new_step_callback: Callable[['BrowserState', 'AgentOutput', int], None] | None = None,
            register_done_callback: Callable[['AgentHistoryList'], None] | None = None,
            tool_calling_method: Optional[str] = 'auto',
            page_extraction_llm: Optional[BaseChatModel] = None,
            planner_llm: Optional[BaseChatModel] = None,
            planner_interval: int = 1,  # Run planner every N steps
            is_hash_deals_agent: bool = False # New flag to indicate Hash Deals Agent

    ):
        # ... (rest of the __init__ method remains the same until system_prompt_class and agent_prompt_class) ...

        if is_hash_deals_agent: # Use HashDeals prompts if it's a Hash Deals Agent
            system_prompt_class = HashDealsSystemPrompt
            agent_prompt_class = HashDealsAgentMessagePrompt
        else: # Otherwise, use the defaults (CustomSystemPrompt, CustomAgentMessagePrompt)
            system_prompt_class = system_prompt_class
            agent_prompt_class = agent_prompt_class


        super().__init__( # Call superclass __init__ with potentially updated prompt classes
            task=task,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            use_vision=use_vision,
            use_vision_for_planner=use_vision_for_planner,
            save_conversation_path=save_conversation_path,
            save_conversation_path_encoding=save_conversation_path_encoding,
            max_failures=max_failures,
            retry_delay=retry_delay,
            system_prompt_class=system_prompt_class, # Use potentially updated system_prompt_class
            agent_prompt_class=agent_prompt_class, # Use potentially updated agent_prompt_class
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            message_context=message_context,
            generate_gif=generate_gif,
            sensitive_data=sensitive_data,
            available_file_paths=available_file_paths,
            include_attributes=include_attributes,
            max_error_length=max_error_length,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content,
            initial_actions=initial_actions,
            register_new_step_callback=register_new_step_callback,
            register_done_callback=register_done_callback,
            tool_calling_method=tool_calling_method,
            planner_llm=planner_llm,
            planner_interval=planner_interval
        )
        # ... (rest of the __init__ method remains the same) ...
