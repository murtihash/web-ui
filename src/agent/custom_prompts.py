import pdb
from typing import List, Optional

from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.browser.views import BrowserState
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime

from .custom_views import CustomAgentStepInfo


class CustomSystemPrompt(SystemPrompt): # Keep the CustomSystemPrompt for general tasks
    def important_rules(self) -> str:
        """
        Returns the important rules for the agent.
        """
        text = r"""
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   ... (rest of the original CustomSystemPrompt rules remain the same) ...
   }
2. ACTIONS: ... (rest of the original CustomSystemPrompt rules remain the same) ...
3. ELEMENT INTERACTION: ... (rest of the original CustomSystemPrompt rules remain the same) ...
4. NAVIGATION & ERROR HANDLING: ... (rest of the original CustomSystemPrompt rules remain the same) ...
5. TASK COMPLETION: ... (rest of the original CustomSystemPrompt rules remain the same) ...
6. VISUAL CONTEXT: ... (rest of the original CustomSystemPrompt rules remain the same) ...
   - Pay special attention to image carousels or sliders that may contain deal information presented visually. Analyze the images in the carousel to identify deals and discounts.
7. Form filling: ... (rest of the original CustomSystemPrompt rules remain the same) ...
8. ACTION SEQUENCING: ... (rest of the original CustomSystemPrompt rules remain the same) ...
9. Extraction: ... (rest of the original CustomSystemPrompt rules remain the same) ...
"""
        text += f"   - use maximum {self.max_actions_per_step} actions per sequence"
        return text

    def input_format(self) -> str: # Keep the input format same
        return """
INPUT STRUCTURE:
... (rest of the original CustomSystemPrompt input_format remains the same) ...
    """


class HashDealsSystemPrompt(SystemPrompt): # New System Prompt for Hash Deals
    def important_rules(self) -> str:
        """
        Returns the important rules specifically for Hash Deals Agent.
        """
        text = r"""
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   {
     "current_state": {
       "deal_evaluation": "Success|Failed|Unknown - Evaluate if deals were successfully found and extracted. Base your evaluation on the current page content and image. Briefly state why/why not.",
       "extracted_deals_summary": "Summarize the deals and discounts you have extracted so far. If no deals are found, output 'No deals found'.",
       "next_steps_deals": "Outline the next steps to find more deals or refine the current search. E.g., 'Check other pages', 'Handle carousel', 'Task Complete' etc.",
       "thought": "Think step-by-step about how to best find and extract all available deals and discounts on this website. Reflect on previous actions and plan the next ones.",
       "summary": "Briefly summarize the next action you are going to take to find more deals."
     },
     "action": [
       * actions in sequences, prioritize actions to find and extract deals. Refer to **Deal-Finding Action Sequences**. Each action MUST be in JSON format: \{action_name\: action_params\}*
     ]
   }

2. ACTIONS: Focus on actions that help find and extract DEALS and DISCOUNTS.

   **Deal-Finding Action Sequences:**
   - Initial Navigation & Age Check: [
       {"go_to_url": {"url": "website_url"}},
       {"handle_age_verification": {}}
     ]
   - Deals Page Navigation: [
       {"navigate_to_deals_page": {}},
       {"extract_deals_information": {}}
     ]
   - Carousel Handling: [
       {"handle_image_carousel_deals": {}},
       {"extract_deals_information": {}}
     ]
   - General Page Extraction (Fallback): [
       {"extract_page_content": {}}
     ]
   - Task Completion: [
       {"done": {"extracted_deals": "Summary of all extracted deals and discounts"}}
     ]

3. WEBSITE NAVIGATION FOR DEALS:
   - Look for "Deals", "Discounts", "Savings", "Promotions", "Offers", "Sales" links in menus, headers, footers, and page body.
   - Prioritize navigation to dedicated deals pages.
   - Use scroll if necessary to find deals sections or links.

4. DEAL EXTRACTION:
   - Focus on extracting structured deal information: Deal Title, Description, Discount (percentage or amount), Price, Original Price, Validity, etc.
   - Extract deals from deal containers, carousels, and general page content.
   - If structured extraction fails, fallback to extracting all relevant text content related to deals.

5. VISUAL CONTEXT FOR DEALS:
   - **Crucially, use the provided screenshot to identify visually prominent deals, especially in image carousels, banners, and promotional sections. Vision is key to finding deals that are not just text-based.**
   - Analyze images for deal text, discount percentages, product information, and calls to action related to deals.

6. AGE VERIFICATION:
   - Always handle age verification prompts at the beginning. Provide age and confirm if necessary using provided details (DOB: 07/25/1994, Name: Mohammad Hashmi).

7. TASK COMPLETION FOR HASH DEALS:
   - Once you have explored all likely sections for deals (deals pages, carousels, main page), and extracted all identifiable deals, use the **Done** action.
   - In the **Done** action, include a comprehensive summary of ALL extracted deals and discounts in the `extracted_deals` parameter. This summary will be the final output for the user.
   - If no deals are found after thorough search, indicate "No deals found" in the `extracted_deals` summary of the Done action.

8. ACTION SEQUENCING FOR EFFICIENCY:
   - Start with navigation to deals pages and carousel handling as these are often primary sources of deals.
   - Use general page content extraction as a fallback to catch any deals not found in dedicated sections.
   - Limit action sequences to a reasonable number to avoid getting stuck.

"""
        text += f"   - use maximum {self.max_actions_per_step} actions per sequence"
        return text

    def input_format(self) -> str: # Keep the input format same
        return """
INPUT STRUCTURE:
... (rest of the original CustomSystemPrompt input_format remains the same) ...
    """


class CustomAgentMessagePrompt(AgentMessagePrompt): # Keep the CustomAgentMessagePrompt for general tasks
    def __init__(
            self,
            state: BrowserState,
            actions: Optional[List[ActionModel]] = None,
            result: Optional[List[ActionResult]] = None,
            include_attributes: list[str] = [],
            max_error_length: int = 400,
            step_info: Optional[CustomAgentStepInfo] = None,
    ):
        super().__init__(state=state,
                         result=result,
                         include_attributes=include_attributes,
                         max_error_length=max_error_length,
                         step_info=step_info
                         )
        self.actions = actions

    def get_user_message(self, use_vision: bool = True) -> HumanMessage: # Keep the get_user_message same
        if self.step_info:
            step_info_description = f'Current step: {self.step_info.step_number}/{self.step_info.max_steps}\n'
        else:
            step_info_description = ''

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        step_info_description += f"Current date and time: {time_str}"

        elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)

        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0

        if elements_text != '':
            if has_content_above:
                elements_text = (
                    f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
                )
            else:
                elements_text = f'[Start of page]\n{elements_text}'
            if has_content_below:
                elements_text = (
                    f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
                )
            else:
                elements_text = f'{elements_text}\n[End of page]'
        else:
            elements_text = 'empty page'

        state_description = f"""
{step_info_description}
1. Task: {self.step_info.task}.
2. Hints(Optional):
{self.step_info.add_infos}
3. Memory:
{self.step_info.memory}
4. Current url: {self.state.url}
5. Available tabs:
{self.state.tabs}
6. Interactive elements:
{elements_text}

**Visual Information:**
- **Examine the screenshot provided. It shows the current visual state of the webpage. Look for any visually presented deals, discounts, or promotions, especially in image carousels or banners. Combine this visual information with the text-based element information to understand the deals available on this page.**
        """

        if self.actions and self.result:
            state_description += "\n **Previous Actions** \n"
            state_description += f'Previous step: {self.step_info.step_number-1}/{self.step_info.max_steps} \n'
            for i, result in enumerate(self.result):
                action = self.actions[i]
                state_description += f"Previous action {i + 1}/{len(self.result)}: {action.model_dump_json(exclude_unset=True)}\n"
                if result.include_in_memory:
                    if result.extracted_content:
                        state_description += f"Result of previous action {i + 1}/{len(self.result)}: {result.extracted_content}\n"
                    if result.error:
                        # only use last 300 characters of error
                        error = result.error[-self.max_error_length:]
                        state_description += (
                            f"Error of previous action {i + 1}/{len(self.result)}: ...{error}\n"
                        )

        if self.state.screenshot and use_vision == True:
            # Format message for vision model
            return HumanMessage(
                content=[
                    {'type': 'text', 'text': state_description},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
                    },
                ]
            )

        return HumanMessage(content=state_description)


class HashDealsAgentMessagePrompt(AgentMessagePrompt): # New Agent Message Prompt for Hash Deals
    def __init__(
            self,
            state: BrowserState,
            actions: Optional[List[ActionModel]] = None,
            result: Optional[List[ActionResult]] = None,
            include_attributes: list[str] = [],
            max_error_length: int = 400,
            step_info: Optional[CustomAgentStepInfo] = None,
    ):
        super().__init__(state=state,
                         result=result,
                         include_attributes=include_attributes,
                         max_error_length=max_error_length,
                         step_info=step_info
                         )
        self.actions = actions

    def get_user_message(self, use_vision: bool = True) -> HumanMessage: # New get_user_message tailored for Hash Deals
        if self.step_info:
            step_info_description = f'Current step: {self.step_info.step_number}/{self.step_info.max_steps} (Finding Deals)\n' # Step description indicates "Finding Deals"
        else:
            step_info_description = ''

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        step_info_description += f"Current date and time: {time_str}"

        elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)

        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0

        if elements_text != '':
            if has_content_above:
                elements_text = (
                    f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
                )
            else:
                elements_text = f'[Start of page]\n{elements_text}'
            if has_content_below:
                elements_text = (
                    f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
                )
            else:
                elements_text = f'{elements_text}\n[End of page]'
        else:
            elements_text = 'empty page'

        state_description = f"""
{step_info_description}
**TASK: Find all available DEALS and DISCOUNTS on this dispensary website.**

**Focus on identifying and extracting any information related to price reductions, special offers, and promotions.**

3. Memory:
{self.step_info.memory}
4. Current url: {self.state.url}
5. Available tabs:
{self.state.tabs}
6. Interactive elements:
{elements_text}

**Visual Information is CRUCIAL for this task:**
- **Actively use the screenshot to look for visual cues about deals.** Dispensary websites often use banners, image carousels, and graphic elements to highlight promotions.
- **Identify deals presented in images within carousels, banners, and other visual sections.**
- **Do not rely solely on text-based elements. Visual analysis is key to comprehensively finding deals.**
        """ # Emphasize visual information even more

        if self.actions and self.result:
            state_description += "\n **Previous Actions & Results** \n" # More descriptive section title
            state_description += f'Previous step: {self.step_info.step_number-1}/{self.step_info.max_steps} \n'
            for i, result in enumerate(self.result):
                action = self.actions[i]
                state_description += f"Previous action {i + 1}/{len(self.result)}: {action.model_dump_json(exclude_unset=True)}\n"
                if result.include_in_memory:
                    if result.extracted_content:
                        state_description += f"Result of previous action {i + 1}/{len(self.result)}: {result.extracted_content}\n"
                    if result.error:
                        # only use last 300 characters of error
                        error = result.error[-self.max_error_length:]
                        state_description += (
                            f"Error of previous action {i + 1}/{len(self.result)}: ...{error}\n"
                        )

        if self.state.screenshot and use_vision == True:
            # Format message for vision model
            return HumanMessage(
                content=[
                    {'type': 'text', 'text': state_description},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
                    },
                ]
            )

        return HumanMessage(content=state_description)
