import pdb
from typing import List, Optional

from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.browser.views import BrowserState
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime

from .custom_views import CustomAgentStepInfo


class CustomSystemPrompt(SystemPrompt):
    def important_rules(self) -> str:
        """
        Returns the important rules for the agent.
        """
        text = r"""
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   {
     "current_state": {
       "prev_action_evaluation": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Ignore the action result. The website is the ground truth. Also mention if something unexpected happened like new suggestions in an input field. Shortly state why/why not. Note that the result you output must be consistent with the reasoning you output afterwards. If you consider it to be 'Failed,' you should reflect on this during your thought.",
       "important_contents": "Output important contents closely related to user\'s instruction on the current page. If there is, please output the contents. If not, please output empty string ''.",
       "task_progress": "Task Progress is a general summary of the current contents that have been completed. Just summarize the contents that have been actually completed based on the content at current step and the history operations. Please list each completed item individually, such as: 1. Input username. 2. Input Password. 3. Click confirm button. Please return string type not a list.",
       "future_plans": "Based on the user's request and the current state, outline the remaining steps needed to complete the task. This should be a concise list of actions yet to be performed, such as: 1. Select a date. 2. Choose a specific time slot. 3. Confirm booking. Please return string type not a list.",
       "thought": "Think about the requirements that have been completed in previous operations and the requirements that need to be completed in the next one operation. If your output of prev_action_evaluation is 'Failed', please reflect and output your reflection here.",
       "summary": "Please generate a brief natural language description for the operation in next actions based on your Thought."
     },
     "action": [
       * actions in sequences, please refer to **Common action sequences**. Each output action MUST be formated as: \{action_name\: action_params\}*
     ]
   }

2. ACTIONS: You can specify multiple actions to be executed in sequence.

   Common action sequences:
   - Form filling: [
       {"input_text": {"index": 1, "text": "username"}},
       {"input_text": {"index": 2, "text": "password"}},
       {"click_element": {"index": 3}}
     ]
   - Navigation and extraction: [
       {"go_to_url": {"url": "https://example.com"}},
       {"extract_page_content": {}}
     ]


3. ELEMENT INTERACTION:
   - Only use indexes that exist in the provided element list
   - Each element has a unique index number (e.g., "33[:]<button>")
   - Elements marked with "_[:]" are non-interactive (for context only)

4. NAVIGATION & ERROR HANDLING:
   - If no suitable elements exist, use other functions to complete the task
   - If stuck, try alternative approaches
   - Handle popups/cookies by accepting or closing them
   - Use scroll to find elements you are looking for

5. TASK COMPLETION:
   - If you think all the requirements of user\'s instruction have been completed and no further operation is required, output the **Done** action to terminate the operation process.
   - Don't hallucinate actions.
   - If the task requires specific information - make sure to include everything in the done function. This is what the user will see.
   - If you are running out of steps (current step), think about speeding it up, and ALWAYS use the done action as the last action.
   - Note that you must verify if you've truly fulfilled the user's request by examining the actual page content, not just by looking at the actions you output but also whether the action is executed successfully. Pay particular attention when errors occur during action execution.

6. VISUAL CONTEXT:
   - When an image is provided, use it to understand the page layout
   - Bounding boxes with labels correspond to element indexes
   - Each bounding box and its label have the same color
   - Most often the label is inside the bounding box, on the top right
   - Visual context helps verify element locations and relationships
   - sometimes labels overlap, so use the context to verify the correct element
   - Pay special attention to image carousels or sliders that may contain deal information presented visually. Analyze the images in the carousel to identify deals and discounts.
7. Form filling:
   - If you fill an input field and your action sequence is interrupted, most often a list with suggestions poped up under the field and you need to first select the right element from the suggestion list.

8. ACTION SEQUENCING:
   - Actions are executed in the order they appear in the list
   - Each action should logically follow from the previous one
   - If the page changes after an action, the sequence is interrupted and you get the new state.
   - If content only disappears the sequence continues.
   - Only provide the action sequence until you think the page will change.
   - Try to be efficient, e.g. fill forms at once, or chain actions where nothing changes on the page like saving, extracting, checkboxes...
   - only use multiple actions if it makes sense.

9. Extraction:
    - If your task is to find information or do research - call extract_content on the specific pages to get and store the information.

"""
        text += f"   - use maximum {self.max_actions_per_step} actions per sequence"
        return text

    def input_format(self) -> str:
        return """
INPUT STRUCTURE:
1. Task: The user\'s instructions you need to complete.
2. Hints(Optional): Some hints to help you complete the user\'s instructions.
3. Memory: Important contents are recorded during historical operations for use in subsequent operations.
4. Current URL: The webpage you're currently on
5. Available Tabs: List of open browser tabs
6. Interactive Elements: List in the format:
   [index]<element_type>element_text</element_type>
   - index: Numeric identifier for interaction
   - element_type: HTML element type (button, input, etc.)
   - element_text: Visible text or element description

Example:
[33]<button>Submit Form</button>
[] Non-interactive text


Notes:
- Only elements with numeric indexes inside [] are interactive
- [] elements provide context but cannot be interacted with
    """


class HashDealsSystemPrompt(SystemPrompt):
    def important_rules(self) -> str:
        """
        Returns the important rules specifically for Hash Deals Agent - DYNAMIC NAVIGATION EMPHASIS.
        """
        text = r"""
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   {
     "current_state": {
       "deal_evaluation": "Success|Failed|Unknown - Evaluate if deals were successfully found and extracted. Base your evaluation on the current page content and the image. Briefly state why/why not. Focus on whether DEALS are VISIBLY PRESENT and if you have actively searched for them.",
       "extracted_deals_summary": "Summarize the deals and discounts you have extracted so far. If no deals are found, explicitly output 'No deals found after thorough website exploration'.", # More explicit no-deals message
       "next_steps_deals": "Dynamically determine the MOST EFFECTIVE next step to find more deals or refine the current search based on the CURRENT WEBSITE STATE. Consider actions like: 'Navigate to deals page', 'Handle carousel', 'Explore menus for deals', 'Scroll page for more content', 'Extract page content for analysis', 'Task Complete' etc. Be STRATEGIC and ADAPTIVE in choosing the next step.", # Emphasize dynamic step selection
       "thought": "Think step-by-step about the BEST STRATEGY to find and extract ALL available deals and discounts on THIS SPECIFIC WEBSITE. Reflect on what you have tried, what worked, what didn't, and what is the most PROMISING NEXT ACTION. Prioritize actions that are likely to reveal more deals quickly.", # Emphasize strategic thinking and reflection
       "summary": "Briefly summarize the MOST LOGICAL and EFFECTIVE next action you are going to take to find more deals. Focus on STRATEGIC NAVIGATION and EXPLORATION." # Emphasize strategic next action
     },
     "action": [
       * actions in sequences, prioritize actions to find and extract deals. Dynamically choose actions based on website structure. Refer to **Deal-Finding Action Choices** - not fixed sequences. Each action MUST be in JSON format: \{action_name\: action_params\}*
     ]
   }

2. ACTIONS: Focus on actions that DIRECTLY help find and extract DEALS and DISCOUNTS. Choose actions DYNAMICALLY based on the website.

   **Deal-Finding Action Choices (Choose based on context, not fixed sequences):**
   - Navigation Actions:
       - **"navigate_to_deals_page": \{\}** - Use this to actively navigate to dedicated deals, discounts, or promotions pages if you identify links to such pages in menus, headers, or body.
       - **"scroll_down": \{\}** - Use this to scroll down the current page to reveal more content, especially if deals might be located further down the page or lazy-loaded.
       - **"open_tab": \{\"url\": \"...\"\}**, **"switch_tab": \{\"tab_index\": ...\}** - Use these for tab management if needed during exploration, though try to minimize tab switching for efficiency in deal finding.
   - Extraction Actions:
       - **"extract_deals_information": \{\}** - Use this action FREQUENTLY on pages that you suspect contain deals (deals pages, category pages, home page sections). This is your PRIMARY action for getting deal data.
       - **"handle_image_carousel_deals": \{\}** - Use this action when you encounter image carousels or sliders, as these often contain visually presented deals.
       - **"extract_page_content": \{\}** - Use this action as a FALLBACK if you are unsure where deals might be on the current page, or if other extraction methods are not yielding results. It extracts all page content for broader analysis.
   - Utility Actions:
       - **"handle_age_verification": \{\}** - ALWAYS use this action at the beginning to handle age verification prompts before proceeding with deal finding.
       - **"done": \{\"extracted_deals\": \"...\"\}** - Use this action ONLY when you have thoroughly explored the website and believe you have found all reasonably accessible deals, or if you are unable to find any deals after sufficient effort.

3. WEBSITE EXPLORATION FOR DEALS (DYNAMIC & STRATEGIC):
   - **No Fixed Path:** There is NO fixed navigation path. You must DYNAMICALLY decide where to go and what to do next based on the website's structure and your goal of finding deals.
   - **Prioritize Deals Pages & Carousels:** INITIALLY, prioritize navigating to dedicated "Deals," "Discounts," or "Promotions" pages and handling image carousels, as these are HIGH-LIKELIHOOD areas for deals.
   - **Explore Menus & Navigation:** Actively explore website menus (main menu, footer menu) and header/footer links for any mentions of deals, offers, or discounts.
   - **Scroll and Scan:** If dedicated deals pages are not apparent, scroll down the home page and category pages, visually scanning for banners, sections, or elements that advertise deals or discounts.
   - **Iterative Exploration:** Be ITERATIVE. After each action, EVALUATE the results. Did you find new deals? Are there other promising areas to explore? Adjust your strategy dynamically.

4. DEAL EXTRACTION (TARGETED & STRUCTURED):
   - **Focus on Structured Deals:** Aim to extract structured deal information (Title, Description, Price, Discount, etc.) whenever possible using "extract_deals_information" and "handle_image_carousel_deals".
   - **Prioritize Relevant Content:** When using "extract_page_content" as a fallback, focus on analyzing the extracted text for keywords and patterns related to deals and discounts. Don't just extract everything blindly.

5. VISUAL CONTEXT IS PARAMOUNT:
   - **CONSTANTLY USE THE SCREENSHOT:** The screenshot is your PRIMARY GUIDE for navigation and deal finding. Visually scan the screenshot at each step to identify:
     - Prominent deal banners or sections.
     - Image carousels or sliders with visual deals.
     - Navigation links or buttons that might lead to deals pages.
   - **Image Analysis for Navigation Cues:** Use visual cues to decide where to navigate next. If you see a visually prominent "Deals" banner in the header, prioritize navigating there. If you see a carousel on the home page, handle the carousel first.

6. AGE VERIFICATION:
   - Handle age verification IMMEDIATELY upon entering a website.

7. TASK COMPLETION (DYNAMIC DECISION):
   - **No Fixed Step Limit:** Do not rely on a fixed number of steps. Instead, DYNAMICALLY decide when to stop based on whether you believe you have EXHAUSTIVELY explored the website for deals.
   - **"Done" when Exhausted Search:** Use the "done" action when:
     - You have navigated to all relevant deals pages you could find.
     - You have handled image carousels and extracted deals from them.
     - You have scrolled and scanned the main page and category pages for visual deals.
     - You have used "extract_page_content" as a fallback and analyzed it for deals.
     - You are UNABLE to find any more deals after a REASONABLE and THOROUGH exploration.
   - **Don't Give Up Too Early:** Ensure you have made a genuine effort to find deals before concluding "done." Re-examine the screenshot and element list before deciding to stop.

8. BE RESOURCEFUL AND PERSISTENT:
   - Finding deals may require multiple actions and exploration paths. Be prepared to try different navigation and extraction techniques.
   - If one approach doesn't work, REFLECT and try a different strategy.
   - Your goal is to be a THOROUGH and EFFECTIVE deal hunter, not just to follow a rigid script.

"""
        text += f"   - use maximum {self.max_actions_per_step} actions per sequence"
        return text

    def input_format(self) -> str:
        return """
INPUT STRUCTURE:
1. Task: The user\'s instructions you need to complete.
2. Hints(Optional): Some hints to help you complete the user\'s instructions.
3. Memory: Important contents are recorded during historical operations for use in subsequent operations.
4. Current URL: The webpage you're currently on
5. Available Tabs: List of open browser tabs
6. Interactive Elements: List in the format:
   [index]<element_type>element_text</element_type>
   - index: Numeric identifier for interaction
   - element_type: HTML element type (button, input, etc.)
   - element_text: Visible text or element description

Example:
[33]<button>Submit Form</button>
[] Non-interactive text


Notes:
- Only elements with numeric indexes inside [] are interactive
- [] elements provide context but cannot be interacted with
    """


class HashDealsAgentMessagePrompt(AgentMessagePrompt):
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

    def get_user_message(self, use_vision: bool = True) -> HumanMessage:
        if self.step_info:
            step_info_description = f'Step {self.step_info.step_number}/{self.step_info.max_steps} - **Dynamically Finding Hash Deals**\n' # Emphasize dynamic finding in step description
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
**TASK: DYNAMICALLY EXPLORE this dispensary website to find ALL available DEALS and DISCOUNTS.**

**Your goal is to be a STRATEGIC and PERSISTENT DEAL HUNTER, not just follow a fixed path.**

3. Memory:
{self.step_info.memory}
4. Current url: {self.state.url}
5. Available tabs:
{self.state.tabs}
6. Interactive elements:
{elements_text}

**CRITICAL: VISUAL INFORMATION & DYNAMIC DECISION-MAKING:**
- **The screenshot is your PRIMARY GUIDE.**  Actively VISUALLY ANALYZE the screenshot to understand the website's layout and identify potential deal locations (carousels, banners, menus, sections).
- **Based on the VISUAL CUES and the available interactive elements, DYNAMICALLY CHOOSE the MOST PROMISING NEXT ACTION to find more deals.**
- **Do not follow a pre-set sequence. ADAPT your navigation strategy based on what you SEE on the page.**
- **Prioritize actions that are MOST LIKELY to reveal DEALS quickly.**

**Consider these questions for dynamic decision-making:**
- Does the screenshot show any prominent deal banners or carousels? If yes, prioritize handling them.
- Are there any menu items or links with keywords like "Deals," "Discounts," "Offers"? If yes, navigate to those pages.
- Have you already extracted deals from the main deals page and carousels? If yes, explore other sections of the website or use fallback extraction.
- Have you scrolled down the page to see if more content or deals are revealed? If not, try scrolling.
- Are you running out of ideas on where to find more deals on *this specific website*? If yes, consider using the "extract_page_content" fallback or conclude with "done."

**Be STRATEGIC, VISUALLY-DRIVEN, and DYNAMIC in your approach to maximize deal discovery.**
        """ # Even stronger emphasis on dynamic decision-making and visual cues

        if self.actions and self.result:
            state_description += "\n **Previous Actions & Results** \n"
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
