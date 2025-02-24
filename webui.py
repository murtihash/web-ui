import pdb
import logging
import json # Import json

from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse
import os

logger = logging.getLogger(__name__)

import gradio as gr

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot
from src.utils.hash_deals_agent import hash_deals_agent # Import hash_deals_agent


# Global variables for persistence
_global_browser = None
_global_browser_context = None
_global_agent = None

# Create the global agent state instance
_global_agent_state = AgentState()

def resolve_sensitive_env_variables(text):
    """
    Replace environment variable placeholders ($SENSITIVE_*) with their values.
    Only replaces variables that start with SENSITIVE_.
    """
    if not text:
        return text

    import re

    # Find all $SENSITIVE_* patterns
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)

    result = text
    for var in env_vars:
        # Remove the $ prefix to get the actual environment variable name
        env_name = var[1:]  # removes the $
        env_value = os.getenv(env_name)
        if env_value is not None:
            # Replace $SENSITIVE_VAR_NAME with its value
            result = result.replace(var, env_value)

    return result

async def stop_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state, _global_browser_context, _global_browser, _global_agent

    try:
        # Request stop
        _global_agent.stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (
            message,                                        # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def stop_research_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state, _global_browser_context, _global_browser

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (                                   # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        is_hash_deals_agent=False
):
    global _global_agent_state
    _global_agent_state.clear_stop()  # Clear any previous stop requests

    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        task = resolve_sensitive_env_variables(task)

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                is_hash_deals_agent=is_hash_deals_agent
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                is_hash_deals_agent=is_hash_deals_agent
            )
        elif agent_type == "hash_deals":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_hash_deals_agent_func(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                is_hash_deals_agent=True
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # ... (rest of run_browser_agent remains same) ...
        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_video,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )


async def run_org_agent( # Keep run_org_agent same
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        is_hash_deals_agent=False
):
    # ... (rest of run_org_agent remains same) ...
    return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file


async def run_custom_agent( # Keep run_custom_agent same
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        is_hash_deals_agent=False
):
    # ... (rest of run_custom_agent remains same) ...
    return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file


async def run_hash_deals_agent_func( # Keep run_hash_deals_agent_func same
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        is_hash_deals_agent=True
):
    # ... (rest of run_hash_deals_agent_func remains same) ...
    history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
    _global_agent.save_history(history_file)

    final_result = history.final_result()
    errors = history.errors()
    model_actions = history.model_actions()
    model_thoughts = history.model_thoughts()

    trace_file = get_latest_files(save_trace_path)

    return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file


async def run_with_stream( # Keep run_with_stream same
    agent_type,
    llm_provider,
    llm_model_name,
    llm_num_ctx,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method
):
    if agent_type == "hash_deals": # No browser view for hash_deals_agent
        result = await run_hash_deals_agent_func( # Call run_hash_deals_agent_func here
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_num_ctx=llm_num_ctx,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            is_hash_deals_agent=True
        )
        yield [gr.update(visible=False)] + list(result) # Hide browser_view for hash_deals_agent
    else: # For other agent types, keep browser view
        # ... (rest of the run_with_stream function remains the same) ...
        yield [
            html_content,
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_videos,
            trace,
            history_file,
            stop_button,
            run_button
        ]


async def close_global_browser(): # Keep close_global_browser same
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None


async def run_hash_deals_agents_ui(website_urls_input, location_names_input, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, headless, disable_security, agent_type):
    """
    Runs hash deals agents for multiple websites in parallel and displays results in UI, including JSON download.
    """
    website_urls = [url.strip() for url in website_urls_input.strip().split('\n') if url.strip()]
    location_names = [name.strip() for name in location_names_input.strip().split('\n') if name.strip()]
    if len(website_urls) != len(location_names):
        return "Error: Number of URLs and Location Names must be the same.", None, None, None # Include None for file

    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        num_ctx=llm_num_ctx,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    deals_results: Dict[str, List[Dict[str, Any]]] = {}
    agent_tasks = []
    for i in range(len(website_urls)):
        url = website_urls[i]
        location = location_names[i]
        task = hash_deals_agent(url, location, llm, headless, disable_security)
        agent_tasks.append(task)
        deals_results[url] = []

    all_deals_lists = await asyncio.gather(*agent_tasks)

    output_markdown = ""
    report_data = {} # Dictionary to hold report data for JSON

    for i in range(len(website_urls)):
        url = website_urls[i]
        location = location_names[i]
        deals_list = all_deals_lists[i]
        report_data[f"{location} - {url}"] = deals_list # Store deals list in report_data

        output_markdown += f"### {location} - [{url}]({url})\n"

        if deals_list:
            for deal in deals_list:
                if "error" in deal:
                    output_markdown += f"- **Error:** {deal['error']} for {deal['website_url']}\n"
                elif "full_page_deals_text" in deal:
                     output_markdown += f"- **Fallback Deals Text (Check Website Manually):**\n   > {deal['full_page_deals_text'][:500]}...\n"
                else:
                    output_markdown += f"- **{deal.get('title', 'No Title')}**\n"
                    if deal.get('description'):
                        output_markdown += f"   - Description: {deal['description']}\n"
                    if deal.get('price'):
                        output_markdown += f"   - Price: **{deal['price']}**"
                        if deal.get('original_price') and deal['original_price'] != "N/A":
                             output_markdown += f" (Original: <del>{deal['original_price']}</del>)\n"
                        else:
                            output_markdown += "\n"
                    else:
                        output_markdown += "\n"
                    output_markdown += "\n"
        else:
            output_markdown += "- No deals information extracted or an error occurred.\n\n"

    # Create JSON report file
    report_json_str = json.dumps(report_data, indent=4)
    report_file_path = "hash_deals_report.json" # Fixed filename for MVP
    with open(report_file_path, 'w') as f:
        f.write(report_json_str)

    return output_markdown, report_file_path, gr.update(value="Stop", interactive=True),  gr.update(interactive=True) # Return file path


def create_ui(config, theme_name="Ocean"):
    # ... (rest of create_ui function remains same) ...
     with gr.TabItem("💰 Hash Deals Agents", id=9): # Hash Deals Agent Tab
                website_urls_input = gr.Textbox(
                    label="Dispensary Website URLs (One per line)",
                    lines=5,
                    placeholder="Enter website URLs...",
                    info="List of dispensary website URLs to scrape for deals."
                )
                location_names_input = gr.Textbox(
                    label="Location Names (Corresponding to URLs)",
                    lines=5,
                    placeholder="Enter location names...",
                    info="List of location names corresponding to each URL."
                )
                with gr.Row():
                    run_hash_deals_button = gr.Button("💰 Run Hash Deals Agents", variant="primary", scale=2)
                    stop_hash_deals_button = gr.Button("⏹️ Stop", variant="stop", scale=1)
                hash_deals_output_display = gr.Markdown(label="Deals and Discounts")
                hash_deals_report_download = gr.File(label="Download Deals Report", visible=True) # Make download button visible


            with gr.TabItem("📊 Results", id=6):
                 # ... (Results Tab - no changes) ...
                 run_hash_deals_button.click(
                    fn=run_hash_deals_agents_ui,
                    inputs=[website_urls_input, location_names_input, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, headless, disable_security, agent_type], # Pass agent_type
                    outputs=[hash_deals_output_display, hash_deals_report_download, stop_hash_deals_button, run_hash_deals_button] # Return file component
                )
                stop_hash_deals_button.click(
                    fn=stop_research_agent, # Using stop_research_agent as it serves the purpose
                    inputs=[],
                    outputs=[stop_hash_deals_button, run_hash_deals_button],
                )
    return demo

def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    config_dict = default_config()

    demo = create_ui(config_dict, theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == '__main__':
    main()
