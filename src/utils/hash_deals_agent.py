import asyncio
import logging
from src.agent.custom_agent import CustomAgent
from src.utils import utils
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.controller.custom_controller import CustomController
from browser_use.browser.browser import BrowserConfig, Browser
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

async def hash_deals_agent(website_url: str, location_name: str, llm, headless: bool = False, disable_security: bool = True) -> List[Dict[str, Any]]:
    """
    Agent to extract hash deals from a given dispensary website.

    Args:
        website_url (str): URL of the dispensary website.
        location_name (str): Name of the location (for context).
        llm: Language model to use.
        headless (bool): Run browser in headless mode.
        disable_security (bool): Disable browser security features.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing structured deal information,
                              or an empty list if no deals are found or an error occurs.
    """
    deals_list: List[Dict[str, Any]] = []
    browser = None
    browser_context = None
    try:
        controller = CustomController()

        @controller.registry.action(
            "Handle age verification if present using provided details."
        )
        async def handle_age_verification(browser: BrowserContext):
            return await controller.registry.actions["Handle age verification if present using provided details."](browser=browser)

        @controller.registry.action(
            "Navigate to deals or discounts page if available."
        )
        async def navigate_to_deals_page(browser: BrowserContext):
            return await controller.registry.actions["Navigate to deals or discounts page if available."](browser=browser)

        @controller.registry.action(
            "Extract deals and discounts information from the current page."
        )
        async def extract_deals_information(browser: BrowserContext):
            return await controller.registry.actions["Extract deals and discounts information from the current page."](browser=browser)


        @controller.registry.action(
            "Handle image carousel for deals by clicking right arrow multiple times."
        )
        async def handle_image_carousel_deals(browser: BrowserContext):
            return await controller.registry.actions["Handle image carousel for deals by clicking right arrow multiple times."](browser=browser)


        browser = Browser(
            config=BrowserConfig(
                headless=headless,
                disable_security=disable_security,
            )
        )
        try: # Added try-except for website unreachability in MVP
            browser_context = await browser.new_context(
                config=BrowserContextConfig(
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(width=1280, height=1080),
                )
            )
        except Exception as website_connect_error: # Handle website unreachability
            error_message = f"Error connecting to website {website_url}: {website_connect_error}"
            logger.error(error_message)
            return [{"error": error_message, "website_url": website_url}] # Return error as structured data

        agent = CustomAgent(
            task=f"Find deals and discounts on the dispensary website for {location_name}.",
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            system_prompt_class=CustomSystemPrompt,
            agent_prompt_class=CustomAgentMessagePrompt,
            max_actions_per_step=5,
        )

        initial_actions = [
            {"go_to_url": {"url": website_url}},
            {"handle_age_verification": {}},
            {"navigate_to_deals_page": {}},
            {"handle_image_carousel_deals": {}}, # Try to handle carousel first
            {"extract_deals_information": {}}, # Then extract all deals
            {"extract_page_content": {}}, # Fallback to extract all page content
            {"done": {}}
        ]
        agent.initial_actions = initial_actions
        history = await agent.run(max_steps=20) # Reduced max steps

        for step_history in history.history:
            for result_item in step_history.result:
                if result_item.structured_content:
                    if isinstance(result_item.structured_content, list):
                        deals_list.extend(result_item.structured_content)
                    else:
                        deals_list.append(result_item.structured_content)


    except Exception as e:
        error_message = f"Error processing {website_url}: {e}"
        logger.error(error_message)
        deals_list.append({"error": error_message, "website_url": website_url})
    finally:
        if browser_context:
            await browser_context.close()
        if browser:
            await browser.close()
    return deals_list
