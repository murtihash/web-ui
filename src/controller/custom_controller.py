import pdb

import pyperclip
from typing import Optional, Type, List, Dict, Any
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging

logger = logging.getLogger(__name__)


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("Paste text from clipboard")
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            # send text to browser
            page = await browser.get_current_page()
            await page.keyboard.type(text)

            return ActionResult(extracted_content=text)

        @self.registry.action(
            "Handle age verification if present using provided details."
        )
        async def handle_age_verification(browser: BrowserContext) -> ActionResult:
            page = await browser.get_current_page()
            try:
                logger.info("Attempting to handle age verification...")
                selectors = ['.age-gate', '#age-verification', '#verify-age-modal', '.age-check']
                confirm_yes_selectors = ['button.verify-yes', 'a.age-gate-confirm', 'button:has-text("Yes, I am")', 'input[type="radio"][value="yes"] + label']
                dob_input_selectors = ['input#age-input', 'input[name="birthdate"]', 'input[type="date"]#dob']
                confirm_dob_button_selectors = ['button#age-verify-submit', 'button:has-text("Submit Age")', '.verify-button']

                age_verification_detected = False
                for selector in selectors:
                    if await page.locator(selector, timeout=5000).count() > 0:
                        age_verification_detected = True
                        logger.info(f"Age verification detected using selector: {selector}")
                        break

                if age_verification_detected:
                    for confirm_selector in confirm_yes_selectors:
                        yes_button = await page.locator(confirm_selector, timeout=5000).first()
                        if await yes_button.count() > 0:
                            await yes_button.click(timeout=10000)
                            logger.info(f"Clicked 'Yes' for age verification using selector: {confirm_selector}")
                            return ActionResult(extracted_content="Clicked 'Yes' for age verification.")

                    for dob_selector in dob_input_selectors:
                        dob_input = await page.locator(dob_selector, timeout=5000).first()
                        if await dob_input.count() > 0:
                            await dob_input.fill('07/25/1994', timeout=10000)
                            logger.info(f"Filled Date of Birth using selector: {dob_selector}")
                            for confirm_button_selector in confirm_dob_button_selectors:
                                confirm_button = await page.locator(confirm_button_selector, timeout=5000).first()
                                if await confirm_button.count() > 0:
                                    await confirm_button.click(timeout=10000)
                                    logger.info(f"Confirmed Date of Birth using selector: {confirm_button_selector}")
                                    return ActionResult(extracted_content="Filled and confirmed Date of Birth.")
                            break
                else:
                    logger.info("No age verification pop-up found.")
                    return ActionResult(extracted_content="No age verification pop-up found.")

            except Exception as e:
                error_msg = f"Error handling age verification: {e}"
                logger.warning(error_msg) # Changed to warning for MVP
                return ActionResult(error=error_msg)

        @self.registry.action(
            "Navigate to deals or discounts page if available."
        )
        async def navigate_to_deals_page(browser: BrowserContext) -> ActionResult:
            page = await browser.get_current_page()
            deal_keywords = ["Deals", "Discounts", "Savings", "Promotions", "Specials", "Offers", "Sales"]
            menu_selectors = ['#main-nav', '.nav-menu', '#top-menu', '.header-navigation', '#menu', '.site-header nav']

            try: # Added try-except for MVP error handling
                for menu_selector in menu_selectors:
                    menu = await page.locator(menu_selector, timeout=5000).first()
                    if await menu.count() > 0:
                        logger.info(f"Searching for deals links in menu: {menu_selector}")
                        for keyword in deal_keywords:
                            deal_link_in_menu = await menu.locator(f'a:has-text("{keyword}")', timeout=3000).first()
                            if await deal_link_in_menu.count() > 0:
                                logger.info(f"Navigating to {keyword} page from menu.")
                                await deal_link_in_menu.click(timeout=10000)
                                await page.wait_for_load_state(timeout=20000)
                                return ActionResult(extracted_content=f"Navigated to {keyword} page from menu.")

                logger.info("Searching body for deals links if not found in menus.")
                for keyword in deal_keywords:
                    deal_link_body = await page.locator(f'a:has-text("{keyword}")', timeout=3000).first()
                    if await deal_link_body.count() > 0:
                        logger.info(f"Navigating to {keyword} page from body.")
                        await deal_link_body.click(timeout=10000)
                        await page.wait_for_load_state(timeout=20000)
                        return ActionResult(extracted_content=f"Navigated to {keyword} page from body.")

                error_msg = "No deals/discounts page link found in menus or body."
                logger.info(error_msg)
                return ActionResult(extracted_content=error_msg)

            except Exception as e: # Basic error handling for MVP
                error_msg = f"Error navigating to deals page: {e}"
                logger.warning(error_msg) # Changed to warning for MVP
                return ActionResult(error=error_msg)


        @self.registry.action(
            "Extract deals and discounts information from the current page."
        )
        async def extract_deals_information(browser: BrowserContext) -> ActionResult:
            page = await browser.get_current_page()
            deals_container_selectors = [
                '#deals-section', '.deals-container', '.discount-offers', '#promotions', '.specials-area',
                '.deals-list', '.discounts-grid', '#offer-items', '.promotion-block', '.shop-deals',
                '#daily-deals', '.weekly-specials', '.featured-deals', '.onsale'
            ]
            deal_item_selectors = ['.deal-item', '.discount-item', '.offer', '.promotion', '.product-card', '.deal-block', '.offer-card']

            extracted_deals: List[Dict[str, Any]] = []

            for container_selector in deals_container_selectors:
                deals_container = await page.locator(container_selector, timeout=5000).first()
                if await deals_container.count() > 0:
                    logger.info(f"Extracting deals from container: {container_selector}")
                    deal_items = await deals_container.locator(','.join(deal_item_selectors), timeout=5000).all()
                    if not deal_items:
                        deal_items = await deals_container.locator('div', timeout=5000).all()

                    for item in deal_items:
                        try: # Added try-except for MVP error handling
                            title_element_text = await item.locator('h2, h3, .deal-title, .discount-title, .offer-title').first().inner_text(timeout=3000)
                            description_element_text = await item.locator('p, .deal-description, .discount-description, .offer-description, .description').first().inner_text(timeout=3000)
                            price_element_text = await item.locator('.price, .deal-price, .discount-price, .offer-price, .current-price, .sale-price').first().inner_text(timeout=3000)
                            original_price_element_text = await item.locator('.original-price, .regular-price, .list-price, .was-price').first().inner_text(timeout=3000)

                            deal_data = {
                                "title": title_element_text.strip() if title_element_text else "No Title",
                                "description": description_element_text.strip() if description_element_text else "No Description",
                                "price": price_element_text.strip() if price_element_text else "Price N/A",
                                "original_price": original_price_element_text.strip() if original_price_element_text else "N/A",
                            }
                            extracted_deals.append(deal_data)
                        except Exception as extract_item_err:
                            logger.warning(f"Error extracting deal item: {extract_item_err}") # Changed to warning for MVP

                    if extracted_deals:
                        return ActionResult(extracted_content=f"Extracted Deals Content from {container_selector}: {extracted_deals}", structured_content=extracted_deals)

            # Fallback extraction remains the same (for MVP, focus on structured extraction first)
            logger.info("No specific deals container found, extracting deals from page body (fallback).")
            body_content = await page.locator('body').inner_text(timeout=10000)
            fallback_deal_data = {"full_page_deals_text": body_content.strip() if body_content else "No Page Content"}
            extracted_deals.append(fallback_deal_data)
            return ActionResult(extracted_content=f"Extracted Deals Content from page body (fallback)", structured_content=extracted_deals)


        @self.registry.action(
            "Handle image carousel for deals by clicking right arrow multiple times."
        )
        async def handle_image_carousel_deals(browser: BrowserContext) -> ActionResult:
            page = await browser.get_current_page()
            carousel_selectors = ['.slick-carousel', '.offer-carousel', '.deals-carousel', '.promotion-slider', '.product-carousel', '.image-slider']
            next_button_selectors = ['.slick-next', '.carousel-next', '.slider-next', 'button.next', '.next-slide', '.carousel-arrow-next']
            deal_item_carousel_selectors = ['.deal-item', '.discount-item', '.offer', '.slide', '.carousel-item', '.product-slide']

            extracted_carousel_deals: List[Dict[str, Any]] = []
            carousel_found = False

            for carousel_selector in carousel_selectors:
                carousel = await page.locator(carousel_selector, timeout=5000).first()
                if await carousel.count() > 0:
                    carousel_found = True
                    logger.info(f"Image carousel detected using selector: {carousel_selector}. Clicking through slides...")

                    for _ in range(5):
                        next_button_clicked = False
                        for next_button_selector in next_button_selectors:
                            carousel_next_button = await carousel.locator(next_button_selector, timeout=3000).first()
                            if await carousel_next_button.count() > 0:
                                try: # Added try-except for MVP error handling
                                    deal_items_carousel = await carousel.locator(','.join(deal_item_carousel_selectors), timeout=5000).all()
                                    for item in deal_items_carousel:
                                        try: # Added try-except for MVP item extraction error handling
                                            title_element_text = await item.locator('h2, h3, .deal-title, .discount-title, .offer-title').first().inner_text(timeout=3000)
                                            description_element_text = await item.locator('p, .deal-description, .discount-description, .offer-description, .description').first().inner_text(timeout=3000)
                                            price_element_text = await item.locator('.price, .deal-price, .discount-price, .offer-price, .current-price, .sale-price').first().inner_text(timeout=3000)
                                            original_price_element_text = await item.locator('.original-price, .regular-price, .list-price, .was-price').first().inner_text(timeout=3000)

                                            deal_data = {
                                                "title": title_element_text.strip() if title_element_text else "No Title",
                                                "description": description_element_text.strip() if description_element_text else "No Description",
                                                "price": price_element_text.strip() if price_element_text else "Price N/A",
                                                "original_price": original_price_element_text.strip() if original_price_element_text else "N/A",
                                            }
                                            extracted_carousel_deals.append(deal_data)
                                        except Exception as extract_carousel_item_err:
                                            logger.warning(f"Error extracting carousel deal item: {extract_carousel_item_err}") # Changed to warning for MVP

                                    await carousel_next_button.click(timeout=10000)
                                    next_button_clicked = True
                                    await asyncio.sleep(1)
                                    break
                                except Exception as carousel_nav_err:
                                    logger.warning(f"Carousel navigation issue or no more slides: {carousel_nav_err}") # Changed to warning for MVP
                                    break
                        if not next_button_clicked:
                            logger.info("No more next buttons found in carousel.")
                            break
                    if extracted_carousel_deals:
                        return ActionResult(extracted_content=f"Extracted deals from carousel: {extracted_carousel_deals}", structured_content=extracted_carousel_deals)
                    else:
                        return ActionResult(extracted_content="No deals extracted from carousel or carousel was empty.")

            if not carousel_found:
                logger.info("No image carousel for deals found.")
                return ActionResult(extracted_content="No image carousel for deals found.")

            return ActionResult(extracted_content="No carousel deals action performed.")
