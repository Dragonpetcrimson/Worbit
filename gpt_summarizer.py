# gpt_summarizer.py - Integrates component relationship information with enhanced privacy
import os
import json
import requests
import re  # Added for regex sanitization
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import traceback
import itertools  # For itertools.chain

# Add tiktoken for accurate token counting
import tiktoken

# Import secure API key handler
from secure_api_key import get_openai_api_key
# Import direct component analyzer
from components.direct_component_analyzer import assign_components_and_relationships
# Import config for limits
from config import Config

# Configuration for OpenAI API
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 2000  # This will be overridden by the dynamic calculation
TEMPERATURE = 0.2

def get_accurate_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Get accurate token count using tiktoken library from OpenAI.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for encoding
        
    Returns:
        Accurate token count
    """
    try:
        # Try to use tiktoken for accurate counting
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except (ImportError, KeyError):
        # Fallback to more conservative estimation if tiktoken isn't available
        # or if the model isn't recognized
        words = text.split()
        # More conservative estimation (1.6x instead of 1.33x)
        return int(len(words) * 1.6)

def get_model_token_limits(model: str) -> int:
    """
    Get the token limit for a specific model.
    
    Args:
        model: Model name
        
    Returns:
        Token limit for the model
    """
    # Define limits for different models
    model_limits = {
        # GPT-3.5 family
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-1106": 16385,
        "gpt-3.5-turbo-0125": 16385,
        
        # GPT-4 family
        "gpt-4": 8192,
        "gpt-4-0613": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-32k-0613": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        
        # Default fallback - conservative estimate
        "default": 8000
    }
    
    # Normalize model name to lowercase
    model_lower = model.lower()
    
    # Return the limit for the specified model, or the default limit if not found
    if model_lower in model_limits:
        return model_limits[model_lower]
    else:
        # For unknown models, check for patterns
        if "gpt-3.5" in model_lower:
            return 16385
        elif "gpt-4-32k" in model_lower:
            return 32768
        elif "gpt-4" in model_lower:
            return 8192
        else:
            return model_limits["default"]

def allocate_token_budgets(model: str, safety_margin: float = 0.2):
    """
    Allocate token budgets based on the model's token limit.
    
    Args:
        model: Model name
        safety_margin: Percentage to reserve for overhead and safety (0.0 to 1.0)
        
    Returns:
        Dictionary of token budgets by section
    """
    # Get the token limit for this model
    model_limit = get_model_token_limits(model)
    
    # Reserve tokens for system message, instructions, and safety margin
    reserved_tokens = int(model_limit * safety_margin) + 500  # 500 tokens for system and instructions
    
    # Calculate available tokens
    available_tokens = model_limit - reserved_tokens
    
    # Allocate tokens proportionally
    return {
        "test_info": int(available_tokens * 0.05),        # 5% for test info
        "component_analysis": int(available_tokens * 0.25),  # 25% for component analysis
        "errors": int(available_tokens * 0.35),              # 35% for errors
        "ocr_data": int(available_tokens * 0.15),            # 15% for OCR
        "scenario_text": int(available_tokens * 0.15),       # 15% for scenario
        "blueprints": int(available_tokens * 0.05)           # 5% for future blueprint data
    }

def reduce_component_analysis(content: Dict, available_tokens: int, reduction_level: int = 0) -> str:
    """
    Reduce component analysis content to fit within token budget.
    
    Args:
        content: Component analysis data
        available_tokens: Available token budget
        reduction_level: Level of reduction (0-3)
        
    Returns:
        Reduced component analysis text
    """
    result = []
    
    # Level 0: Full content (no reduction)
    if reduction_level == 0:
        # Include primary component information
        primary = content.get("primary_issue_component", "unknown")
        if primary != "unknown":
            result.append(f"PRIMARY ISSUE COMPONENT: {primary.upper()}")
            
            # Find component info
            primary_comp_info = next((c for c in content.get("component_summary", []) 
                                     if c.get("id") == primary), {})
            
            if primary_comp_info:
                result.append(f"DESCRIPTION: {primary_comp_info.get('description', '')}")
                result.append(f"ERROR COUNT: {primary_comp_info.get('error_count', 0)}")
        
        # Add affected components (all)
        result.append("\nAFFECTED COMPONENTS:")
        for comp in content.get("component_summary", []):
            if comp.get("id") != primary:
                result.append(f"- {comp.get('name', comp.get('id', '').upper())}: "
                             f"{comp.get('error_count', 0)} errors - {comp.get('description', '')}")
    
    # Level 1: Reduced content
    elif reduction_level == 1:
        # Include primary component information
        primary = content.get("primary_issue_component", "unknown")
        if primary != "unknown":
            result.append(f"PRIMARY ISSUE COMPONENT: {primary.upper()}")
            
            # Find component info (abbreviated)
            primary_comp_info = next((c for c in content.get("component_summary", []) 
                                     if c.get("id") == primary), {})
            
            if primary_comp_info:
                result.append(f"DESCRIPTION: {primary_comp_info.get('description', '')[:150]}")
                result.append(f"ERROR COUNT: {primary_comp_info.get('error_count', 0)}")
        
        # Add affected components (limited to 5)
        result.append("\nAFFECTED COMPONENTS:")
        added = 0
        for comp in content.get("component_summary", []):
            if comp.get("id") != primary and added < 5:
                result.append(f"- {comp.get('name', comp.get('id', '').upper())}: "
                             f"{comp.get('error_count', 0)} errors")
                added += 1
    
    # Level 2: Minimal content
    elif reduction_level == 2:
        # Include primary component only
        primary = content.get("primary_issue_component", "unknown")
        if primary != "unknown":
            result.append(f"PRIMARY ISSUE COMPONENT: {primary.upper()}")
            result.append(f"ERROR COUNT: {content.get('component_error_counts', {}).get(primary, 0)}")
        
        # Add affected components (count only)
        affected = content.get("components_with_issues", [])
        if affected:
            result.append(f"AFFECTED COMPONENTS: {len(affected)} components")
    
    # Level 3: Minimal identification only
    else:
        primary = content.get("primary_issue_component", "unknown")
        if primary != "unknown":
            result.append(f"PRIMARY ISSUE COMPONENT: {primary.upper()}")
    
    # Join all lines
    result_text = "\n".join(result)
    
    # Check if we're still over budget
    if get_accurate_token_count(result_text) > available_tokens:
        # If we're already at maximum reduction, truncate
        if reduction_level >= 3:
            # Just keep the first part until we fit
            return result_text[:int(available_tokens * 0.75)]
        else:
            # Try next reduction level
            return reduce_component_analysis(content, available_tokens, reduction_level + 1)
    
    return result_text

def reduce_errors(errors: List[Dict], available_tokens: int, reduction_level: int = 0) -> str:
    """
    Reduce error content to fit within token budget.
    
    Args:
        errors: List of errors
        available_tokens: Available token budget
        reduction_level: Level of reduction (0-3)
        
    Returns:
        Reduced errors text
    """
    result = []
    
    # Sort errors by severity
    sorted_errors = sorted(
        errors,
        key=lambda e: {'High': 0, 'Medium': 1, 'Low': 2}.get(e.get('severity', 'Low'), 3)
    )
    
    # Level 0: Include all high and medium severity errors
    if reduction_level == 0:
        result.append("## Key Errors")
        
        high_errors = [e for e in sorted_errors if e.get('severity') == 'High']
        medium_errors = [e for e in sorted_errors if e.get('severity') == 'Medium']
        
        if high_errors:
            result.append("\n### High Severity Errors")
            for error in high_errors:
                component = error.get('component', 'unknown')
                text = sanitize_text_for_api(error.get('text', '')[:200])
                result.append(f"- [{component.upper()}] {text}")
        
        if medium_errors:
            result.append("\n### Medium Severity Errors")
            for i, error in enumerate(medium_errors):
                if i >= 5:  # Limit to 5 medium errors
                    result.append(f"- ... and {len(medium_errors) - 5} more medium severity errors")
                    break
                component = error.get('component', 'unknown')
                text = sanitize_text_for_api(error.get('text', '')[:150])
                result.append(f"- [{component.upper()}] {text}")
    
    # Level 1: Include only high severity and top 3 medium severity
    elif reduction_level == 1:
        result.append("## Key Errors")
        
        high_errors = [e for e in sorted_errors if e.get('severity') == 'High']
        medium_errors = [e for e in sorted_errors if e.get('severity') == 'Medium']
        
        if high_errors:
            result.append("\n### High Severity Errors")
            for i, error in enumerate(high_errors):
                if i >= 5:  # Limit to 5 high errors
                    result.append(f"- ... and {len(high_errors) - 5} more high severity errors")
                    break
                component = error.get('component', 'unknown')
                text = sanitize_text_for_api(error.get('text', '')[:150])
                result.append(f"- [{component.upper()}] {text}")
        
        if medium_errors:
            result.append("\n### Medium Severity Errors")
            for i, error in enumerate(medium_errors):
                if i >= 3:  # Limit to 3 medium errors
                    result.append(f"- ... and {len(medium_errors) - 3} more medium severity errors")
                    break
                component = error.get('component', 'unknown')
                text = sanitize_text_for_api(error.get('text', '')[:100])
                result.append(f"- [{component.upper()}] {text}")
    
    # Level 2: Include only high severity errors
    elif reduction_level == 2:
        result.append("## Key Errors")
        
        high_errors = [e for e in sorted_errors if e.get('severity') == 'High']
        
        if high_errors:
            for i, error in enumerate(high_errors):
                if i >= 3:  # Limit to 3 high errors
                    result.append(f"- ... and {len(high_errors) - 3} more high severity errors")
                    break
                component = error.get('component', 'unknown')
                text = sanitize_text_for_api(error.get('text', '')[:100])
                result.append(f"- [{component.upper()}] {text}")
        
        # Just counts for other severities
        medium_errors = [e for e in sorted_errors if e.get('severity') == 'Medium']
        low_errors = [e for e in sorted_errors if e.get('severity') == 'Low']
        
        if medium_errors:
            result.append(f"- {len(medium_errors)} medium severity errors")
        if low_errors:
            result.append(f"- {len(low_errors)} low severity errors")
    
    # Level 3: Include only the top high severity error
    else:
        result.append("## Critical Error")
        
        high_errors = [e for e in sorted_errors if e.get('severity') == 'High']
        
        if high_errors:
            error = high_errors[0]
            component = error.get('component', 'unknown')
            text = sanitize_text_for_api(error.get('text', '')[:100])
            result.append(f"- [{component.upper()}] {text}")
        
        # Just total count
        result.append(f"- Total: {len(errors)} errors across all severities")
    
    # Join all lines
    result_text = "\n".join(result)
    
    # Check if we're still over budget
    if get_accurate_token_count(result_text) > available_tokens:
        # If we're already at maximum reduction, truncate
        if reduction_level >= 3:
            # Just keep the first part until we fit
            return result_text[:int(available_tokens * 0.75)]
        else:
            # Try next reduction level
            return reduce_errors(errors, available_tokens, reduction_level + 1)
    
    return result_text

def reduce_ocr_data(ocr_data: List[Dict], available_tokens: int, reduction_level: int = 0) -> str:
    """
    Reduce OCR data to fit within token budget.
    
    Args:
        ocr_data: List of OCR entries
        available_tokens: Available token budget
        reduction_level: Level of reduction (0-3)
        
    Returns:
        Reduced OCR text
    """
    if not ocr_data:
        return ""
    
    result = ["## OCR-Extracted Text from Screenshots:"]
    
    # Level 0: Include all OCR data (up to 5 entries)
    if reduction_level == 0:
        max_entries = 5
        max_length = 200
    # Level 1: Include fewer entries with shorter text
    elif reduction_level == 1:
        max_entries = 3
        max_length = 150
    # Level 2: Include minimal entries
    elif reduction_level == 2:
        max_entries = 2
        max_length = 100
    # Level 3: Include just one entry
    else:
        max_entries = 1
        max_length = 75
    
    # Add entries
    for i, entry in enumerate(ocr_data):
        if i >= max_entries:
            result.append(f"- ... and {len(ocr_data) - max_entries} more screenshots")
            break
        
        ocr_text = entry.get('text', '')[:max_length]
        sanitized_ocr = sanitize_text_for_api(ocr_text)
        result.append(f"- Screenshot ({entry['file']}): {sanitized_ocr}{'...' if len(entry['text']) > max_length else ''}")
    
    # Join all lines
    result_text = "\n".join(result)
    
    # Check if we're still over budget
    if get_accurate_token_count(result_text) > available_tokens:
        # If we're already at maximum reduction, truncate
        if reduction_level >= 3:
            # Just keep the first part until we fit
            return result_text[:int(available_tokens * 0.75)]
        else:
            # Try next reduction level
            return reduce_ocr_data(ocr_data, available_tokens, reduction_level + 1)
    
    return result_text

def reduce_scenario_text(scenario_text: str, available_tokens: int, reduction_level: int = 0) -> str:
    """
    Reduce scenario text to fit within token budget.
    
    Args:
        scenario_text: Scenario description text
        available_tokens: Available token budget
        reduction_level: Level of reduction (0-3)
        
    Returns:
        Reduced scenario text
    """
    if not scenario_text:
        return ""
    
    result = ["## Test Scenario"]
    
    # Level 0: Include full scenario text (up to 500 chars)
    if reduction_level == 0:
        max_length = 500
    # Level 1: Include partial scenario text
    elif reduction_level == 1:
        max_length = 300
    # Level 2: Include minimal scenario text
    elif reduction_level == 2:
        max_length = 200
    # Level 3: Include just the beginning
    else:
        max_length = 100
    
    # Sanitize and limit scenario text
    sanitized_scenario = sanitize_text_for_api(scenario_text[:max_length])
    result.append(f"{sanitized_scenario}{'...' if len(scenario_text) > max_length else ''}")
    
    # Join all lines
    result_text = "\n".join(result)
    
    # Check if we're still over budget
    if get_accurate_token_count(result_text) > available_tokens:
        # If we're already at maximum reduction, truncate
        if reduction_level >= 3:
            # Just keep the first part until we fit
            return result_text[:int(available_tokens * 0.75)]
        else:
            # Try next reduction level
            return reduce_scenario_text(scenario_text, available_tokens, reduction_level + 1)
    
    return result_text

def build_token_managed_prompt(test_id: str, all_data: Dict, model: str) -> str:
    """
    Build a prompt with dynamic token management based on model limits.
    
    Args:
        test_id: Test ID
        all_data: All data for the prompt
        model: Model to use
        
    Returns:
        Prompt text within token limits
    """
    # Get model token limit
    model_limit = get_model_token_limits(model)
    logging.info(f"Building prompt for model {model} with token limit {model_limit}")
    
    # Reserve 25% for response
    response_tokens = int(model_limit * 0.25)
    input_token_limit = model_limit - response_tokens
    
    # Allocate budgets
    budgets = allocate_token_budgets(model)
    
    # Start with the base prompt text
    base_prompt = f"""You are a test automation expert analyzing software test failures. Your goal is to identify the ROOT CAUSE of the test failure and provide ACTIONABLE STEPS.

Test ID: {test_id}

"""
    base_tokens = get_accurate_token_count(base_prompt, model)
    remaining_tokens = input_token_limit - base_tokens
    
    # Try to build the prompt with progressive reduction
    sections = {}
    reduction_level = 0
    
    # Keep trying different reduction levels until we fit
    while reduction_level <= 3:
        logging.info(f"Attempting to build prompt with reduction level {reduction_level}")
        try:
            # Build each section with current reduction level
            section_texts = {}
            
            # Component analysis section
            if "component_analysis" in all_data and all_data["component_analysis"]:
                section_texts["component_analysis"] = reduce_component_analysis(
                    all_data["component_analysis"],
                    budgets["component_analysis"],
                    reduction_level
                )
            
            # Errors section
            if "errors" in all_data and all_data["errors"]:
                section_texts["errors"] = reduce_errors(
                    all_data["errors"],
                    budgets["errors"],
                    reduction_level
                )
                
            # OCR data section
            if "ocr_data" in all_data and all_data["ocr_data"]:
                section_texts["ocr_data"] = reduce_ocr_data(
                    all_data["ocr_data"],
                    budgets["ocr_data"],
                    reduction_level
                )
                
            # Scenario text section
            if "scenario_text" in all_data and all_data["scenario_text"]:
                section_texts["scenario_text"] = reduce_scenario_text(
                    all_data["scenario_text"],
                    budgets["scenario_text"],
                    reduction_level
                )
            
            # Combine all sections
            full_prompt = base_prompt
            for section_name, section_text in section_texts.items():
                if section_text:
                    full_prompt += section_text + "\n\n"
            
            # Add analysis instructions
            full_prompt += """Please provide a CONCISE analysis in the following format:

1. ROOT CAUSE: [Identify the fundamental issue causing the test failure, clearly stating the component responsible]

2. IMPACT: [Briefly explain what functionality is affected]

3. RECOMMENDED ACTIONS: [2-3 bullet points of specific actions to resolve the issue, focusing on the primary component]
"""
            
            # Check final token count
            final_tokens = get_accurate_token_count(full_prompt, model)
            if final_tokens <= input_token_limit:
                # Success! We have a prompt within the limit
                logging.info(f"Successfully built prompt with {final_tokens} tokens (limit: {input_token_limit})")
                return full_prompt
            else:
                # Try next reduction level
                logging.warning(f"Prompt too large ({final_tokens} tokens). Trying next reduction level.")
                reduction_level += 1
                
        except Exception as e:
            logging.error(f"Error building prompt at reduction level {reduction_level}: {str(e)}")
            traceback.print_exc()
            reduction_level += 1
    
    # If we couldn't fit even with maximum reduction, use fallback
    logging.error("Could not build prompt within token limits even with maximum reduction")
    return build_minimal_fallback_prompt(test_id, all_data)

def build_minimal_fallback_prompt(test_id: str, all_data: Dict) -> str:
    """
    Build a minimal fallback prompt for extreme cases.
    
    Args:
        test_id: Test ID
        all_data: All data
        
    Returns:
        Minimal prompt
    """
    # Extract critical information only
    primary_component = "unknown"
    if "component_analysis" in all_data and all_data["component_analysis"]:
        primary_component = all_data["component_analysis"].get("primary_issue_component", "unknown")
    
    # Get the most severe error
    critical_error = ""
    if "errors" in all_data and all_data["errors"]:
        sorted_errors = sorted(
            all_data["errors"],
            key=lambda e: {'High': 0, 'Medium': 1, 'Low': 2}.get(e.get('severity', 'Low'), 3)
        )
        if sorted_errors:
            error_text = sorted_errors[0].get('text', '')[:100]
            critical_error = sanitize_text_for_api(error_text)
    
    # Build minimal prompt
    return f"""You are a test automation expert analyzing a test failure. Provide a brief analysis.

Test ID: {test_id}
Primary Component: {primary_component.upper()}
Critical Error: {critical_error}

Please provide a CONCISE analysis of what might have gone wrong and what actions to take.
"""

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    This is a simple approximation based on whitespace tokenization.
    
    Args:
        text: The text to estimate token count for
        
    Returns:
        Estimated token count
    """
    # A simple approximation: 1 token ≈ 0.75 words
    # This is conservative to avoid underestimation
    words = text.split()
    return max(1, int(len(words) * 1.33))

def build_clustered_prompt(test_id: str, clusters: Dict[int, List[Dict]], ocr_data: List[Dict], 
                          scenario_text: str = "", limited: bool = False) -> str:
    """
    Build a prompt for GPT with clustered error information, OCR data, and scenario context.
    
    Args:
        test_id: Test ID
        clusters: Dictionary mapping cluster IDs to lists of errors
        ocr_data: OCR data extracted from images
        scenario_text: Feature file scenario text (optional)
        limited: Whether to use a shorter prompt for token limits
        
    Returns:
        Formatted prompt for GPT
    """
    prompt = f"""You are a test automation expert analyzing software test failures. Your goal is to identify the ROOT CAUSE of the test failure and provide ACTIONABLE STEPS.

Test ID: {test_id}

"""

    # Add scenario context if available
    if scenario_text:
        # Sanitize scenario text to remove potential PII
        sanitized_scenario = sanitize_text_for_api(scenario_text[:500])
        prompt += "## Test Scenario\n"
        prompt += f"{sanitized_scenario}{'...' if len(scenario_text) > 500 else ''}\n\n"

    # Add clustered errors (limited to save tokens if needed)
    prompt += "## Key Errors Grouped by Similarity\n\n"
    max_clusters = 3 if limited else 5
    max_errors_per_cluster = 2 if limited else 3
    
    sorted_clusters = sorted(
        clusters.items(),
        key=lambda x: sum(1 for e in x[1] if e.get('severity') == 'High'),
        reverse=True
    )
    
    for i, (cluster_id, errors) in enumerate(sorted_clusters):
        if i >= max_clusters:
            break
            
        high_count = sum(1 for e in errors if e.get('severity') == 'High')
        medium_count = sum(1 for e in errors if e.get('severity') == 'Medium')
        low_count = sum(1 for e in errors if e.get('severity') == 'Low')
        
        prompt += f"### Cluster {cluster_id} ({len(errors)} errors: {high_count} High, {medium_count} Medium, {low_count} Low)\n"
        
        # Sort errors by severity
        sorted_errors = sorted(
            errors,
            key=lambda e: {'High': 0, 'Medium': 1, 'Low': 2}.get(e.get('severity', 'Low'), 3)
        )
        
        for j, error in enumerate(sorted_errors):
            if j >= max_errors_per_cluster:
                break
                
            # Sanitize error text to remove potential PII
            error_text = error.get('text', '')
            sanitized_text = sanitize_text_for_api(error_text)
            
            # Add component info if available
            component_info = ""
            if 'component' in error and error['component'] != 'unknown':
                component_info = f" [Component: {error['component'].upper()}]"
                
            prompt += f"- Severity: {error['severity']}{component_info} - {sanitized_text}\n"
            
        if len(errors) > max_errors_per_cluster:
            prompt += f"- ... and {len(errors) - max_errors_per_cluster} more similar errors\n"
            
        prompt += "\n"

    # Add OCR data if available (limited)
    if ocr_data:
        prompt += "## OCR-Extracted Text from Screenshots:\n"
        max_ocr = 2 if limited else 4
        for i, entry in enumerate(ocr_data):
            if i >= max_ocr:
                break
            # Sanitize OCR text as well
            ocr_text = entry.get('text', '')[:150]
            sanitized_ocr = sanitize_text_for_api(ocr_text)
            prompt += f"- Screenshot ({entry['file']}): {sanitized_ocr}{'...' if len(entry['text']) > 150 else ''}\n"
        if len(ocr_data) > max_ocr:
            prompt += f"- ... and {len(ocr_data) - max_ocr} more screenshots\n"
        prompt += "\n"

    # Add expected output format
    prompt += """Please provide a CONCISE analysis in the following format:
1. ROOT CAUSE: [Identify the fundamental issue causing the test failure]
2. IMPACT: [Briefly explain what functionality is affected]
3. RECOMMENDED ACTIONS: [2-3 bullet points of specific actions to resolve the issue]
"""

    return prompt

def build_gpt_prompt(test_id: str, errors: List[Dict], component_summary: List[Dict], 
                    primary_issue_component: str, clusters: Dict[int, List[Dict]] = None,
                    ocr_data: List[Dict] = None, scenario_text: str = "",
                    limited: bool = False) -> str:
    """
    Build an enhanced GPT prompt with component relationship information.
    
    Args:
        test_id: Test ID
        errors: List of error dictionaries
        component_summary: Summary of components involved
        primary_issue_component: The component identified as root cause
        clusters: Dictionary mapping cluster IDs to lists of errors (optional)
        ocr_data: OCR data extracted from images (optional)
        scenario_text: Feature file scenario text (optional)
        limited: Whether to use a shorter prompt for token limits
        
    Returns:
        Enhanced prompt for GPT
    """
    prompt = f"""You are a test automation expert analyzing software test failures. Your goal is to identify the ROOT CAUSE of the test failure and provide ACTIONABLE STEPS.

Test ID: {test_id}

"""

    # Add component information
    prompt += "## Component Analysis\n\n"
    
    if primary_issue_component != 'unknown':
        # Find the primary component info
        primary_comp_info = next((c for c in component_summary if c["id"] == primary_issue_component), {})
        
        prompt += f"PRIMARY ISSUE COMPONENT: {primary_comp_info.get('name', primary_issue_component.upper())}\n"
        prompt += f"DESCRIPTION: {primary_comp_info.get('description', '')}\n"
        prompt += f"ERROR COUNT: {primary_comp_info.get('error_count', 0)}\n\n"
        
        # Add affected components
        prompt += "AFFECTED COMPONENTS:\n"
        for comp in component_summary:
            if comp["id"] in primary_comp_info.get("related_to", []):
                prompt += f"- {comp['name']}: {comp['error_count']} errors - {comp['description']}\n"
        
        prompt += "\n"
    
    # Add scenario context if available
    if scenario_text:
        # Sanitize scenario text to remove potential PII
        sanitized_scenario = sanitize_text_for_api(scenario_text[:500])
        prompt += "## Test Scenario\n"
        prompt += f"{sanitized_scenario}{'...' if len(scenario_text) > 500 else ''}\n\n"

    # Add clustered errors
    prompt += "## Key Errors Grouped by Similarity\n\n"
    
    if clusters:
        max_clusters = 3 if limited else 5
        max_errors_per_cluster = 2 if limited else 3
        
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: sum(1 for e in x[1] if e.get('severity') == 'High'),
            reverse=True
        )
        
        for i, (cluster_id, cluster_errors) in enumerate(sorted_clusters):
            if i >= max_clusters:
                break
                
            high_count = sum(1 for e in cluster_errors if e.get('severity') == 'High')
            medium_count = sum(1 for e in cluster_errors if e.get('severity') == 'Medium')
            low_count = sum(1 for e in cluster_errors if e.get('severity') == 'Low')
            
            prompt += f"### Cluster {cluster_id} ({len(cluster_errors)} errors: {high_count} High, {medium_count} Medium, {low_count} Low)\n"
            
            # Group by component within cluster
            comp_errors = {}
            for error in cluster_errors:
                comp = error.get('component', 'unknown')
                if comp not in comp_errors:
                    comp_errors[comp] = []
                comp_errors[comp].append(error)
            
            # List errors by component
            for comp, errors in comp_errors.items():
                comp_info = next((c for c in component_summary if c["id"] == comp), {"name": comp.upper()})
                prompt += f"#### {comp_info.get('name', comp.upper())} ({len(errors)} errors)\n\n"
                
                # List first few errors from this component
                sorted_errors = sorted(
                    errors,
                    key=lambda e: {'High': 0, 'Medium': 1, 'Low': 2}.get(e.get('severity', 'Low'), 3)
                )
                
                for j, error in enumerate(sorted_errors):
                    if j >= max_errors_per_cluster:
                        break
                        
                    # Sanitize error text to remove potential PII
                    error_text = error.get('text', '')
                    sanitized_text = sanitize_text_for_api(error_text)
                    prompt += f"- Severity: {error.get('severity', 'Unknown')} - {sanitized_text}\n"
                
                if len(errors) > max_errors_per_cluster:
                    prompt += f"- ... and {len(errors) - max_errors_per_cluster} more similar errors\n"
                
                prompt += "\n"
    else:
        # If no clusters, just list some errors
        max_errors = 5 if limited else 10
        sorted_errors = sorted(
            errors[:max_errors],
            key=lambda e: {'High': 0, 'Medium': 1, 'Low': 2}.get(e.get('severity', 'Low'), 3)
        )
        
        for error in sorted_errors:
            # Sanitize error text to remove potential PII
            error_text = error.get('text', '')
            sanitized_text = sanitize_text_for_api(error_text)
            
            component = error.get('component', 'unknown')
            component_tag = f" [{component.upper()}]" if component != 'unknown' else ""
            prompt += f"- Severity: {error.get('severity', 'Unknown')}{component_tag} - {sanitized_text}\n"
        
        prompt += "\n"

    # Add OCR data if available (limited)
    if ocr_data:
        prompt += "## OCR-Extracted Text from Screenshots:\n"
        max_ocr = 2 if limited else 4
        for i, entry in enumerate(ocr_data):
            if i >= max_ocr:
                break
            # Sanitize OCR text as well
            ocr_text = entry.get('text', '')[:150]
            sanitized_ocr = sanitize_text_for_api(ocr_text)
            prompt += f"- Screenshot ({entry['file']}): {sanitized_ocr}{'...' if len(entry['text']) > 150 else ''}\n"
        if len(ocr_data) > max_ocr:
            prompt += f"- ... and {len(ocr_data) - max_ocr} more screenshots\n"
        prompt += "\n"

    # Add analysis instructions for GPT
    prompt += """Please provide a CONCISE analysis in the following format:

1. ROOT CAUSE: [Identify the fundamental issue causing the test failure, clearly stating the component responsible]

2. COMPONENT RELATIONSHIPS: [Explain how the issues in the primary component might have affected other components. How do these components interact?]

3. IMPACT: [Briefly explain what functionality is affected]

4. RECOMMENDED ACTIONS: [2-3 bullet points of specific actions to resolve the issue, focusing on the primary component]

5. SUMMARY: [Provide a 2-3 sentence summary for a bug report, clearly identifying the component responsible for the failure]
"""
    
    return prompt

def sanitize_text_for_api(text: str) -> str:
    """
    Sanitize text before sending to API to remove potentially sensitive information.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text string
    """
    if not isinstance(text, str):
        text = str(text)
        
    # Replace potential sensitive information
    # Replace IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_ADDRESS]', text)
    
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Replace full file paths that might contain usernames
    text = re.sub(r'C:\\Users\\[^\\]+\\', r'C:\\Users\\[USER]\\', text)
    text = re.sub(r'/home/[^/]+/', '/home/[USER]/', text)
    
    # Obscure potential API keys and tokens
    text = re.sub(r'(?i)(api[_-]?key|token|secret|password|pwd)(["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-\.]{20,})', 
                 r'\1\2[REDACTED]', text)
    
    return text

def enhance_prompt_with_component_data(prompt: str, component_analysis: Dict[str, Any]) -> str:
    """
    Enhance the GPT prompt with component relationship data.
    
    Args:
        prompt: Original GPT prompt
        component_analysis: Component analysis results
        
    Returns:
        Enhanced prompt with component information
    """
    # Skip if no component analysis
    if not component_analysis:
        return prompt
        
    try:
        # Get component analysis data
        enhanced_data = {}
        
        # Try to use the component integration module if available
        try:
            from components.component_integration import ComponentIntegration
            component_schema_path = os.path.join('components', 'schemas', 'component_schema.json')
            
            if os.path.exists(component_schema_path):
                integrator = ComponentIntegration(component_schema_path)
                enhanced_data = integrator.get_enhanced_report_data(component_analysis)
        except ImportError:
            # Manually extract basic information if integration module not available
            enhanced_data = {
                "component_analysis": {
                    "root_cause": {},
                    "affected_components": []
                },
                "root_cause_errors": []
            }
            
            # Try to get component error analysis
            analysis_path = component_analysis.get("analysis_files", {}).get("component_analysis")
            if analysis_path and os.path.exists(analysis_path):
                try:
                    with open(analysis_path, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                        
                        # Extract root cause component info
                        root_cause_id = analysis.get("root_cause_component")
                        if root_cause_id and root_cause_id != "unknown":
                            error_count = analysis.get("component_error_counts", {}).get(root_cause_id, 0)
                            enhanced_data["component_analysis"]["root_cause"] = {
                                "id": root_cause_id,
                                "name": root_cause_id.upper(),
                                "error_count": error_count
                            }
                        
                        # Extract affected components
                        for comp_id in analysis.get("components_with_issues", []):
                            if comp_id != "unknown":
                                error_count = analysis.get("component_error_counts", {}).get(comp_id, 0)
                                enhanced_data["component_analysis"]["affected_components"].append({
                                    "id": comp_id,
                                    "name": comp_id.upper(),
                                    "error_count": error_count
                                })
                except Exception as e:
                    logging.warning(f"Error loading component analysis data: {str(e)}")
            
            # Try to get enhanced clustering data
            clusters_path = component_analysis.get("analysis_files", {}).get("enhanced_clusters")
            if clusters_path and os.path.exists(clusters_path):
                try:
                    with open(clusters_path, 'r', encoding='utf-8') as f:
                        clusters_data = json.load(f)
                        enhanced_data["root_cause_errors"] = clusters_data.get("root_cause_errors", [])[:3]
                except Exception as e:
                    logging.warning(f"Error loading enhanced clusters data: {str(e)}")
        
        # Add component information to the prompt
        if enhanced_data.get("component_analysis", {}).get("root_cause"):
            root_cause = enhanced_data["component_analysis"]["root_cause"]
            prompt += f"\n\nROOT CAUSE COMPONENT: {root_cause.get('name', 'Unknown')} - {root_cause.get('description', 'Component in the system')}\n"
            prompt += f"This component had {root_cause.get('error_count', 0)} errors during the test.\n"
        
        # Add affected components
        if enhanced_data.get("component_analysis", {}).get("affected_components"):
            prompt += "\nAFFECTED COMPONENTS:\n"
            for component in enhanced_data["component_analysis"]["affected_components"][:5]:
                prompt += f"- {component.get('name', 'Unknown')} ({component.get('error_count', 0)} errors)\n"
        
        # Add root cause errors
        if enhanced_data.get("root_cause_errors"):
            prompt += "\nROOT CAUSE ERRORS:\n"
            for error in enhanced_data["root_cause_errors"][:3]:
                severity = error.get("severity", "Unknown")
                component = error.get("component", "unknown")
                text = error.get("text", "")
                # Sanitize error text
                sanitized_text = sanitize_text_for_api(text[:150])
                prompt += f"- [{severity}] {component.upper()}: {sanitized_text}{'...' if len(text) > 150 else ''}\n"
        
        # Add error propagation paths
        if enhanced_data.get("causal_chains"):
            prompt += "\nLIKELY ERROR PROPAGATION:\n"
            for i, chain in enumerate(enhanced_data["causal_chains"][:1]):
                path = " → ".join([err.get("component", "unknown").upper() for err in chain])
                prompt += f"Error propagation path: {path}\n"
        
        logging.info("Enhanced GPT prompt with component analysis data")
        
    except Exception as e:
        logging.warning(f"Error enhancing prompt with component data: {str(e)}")
        traceback.print_exc()
    
    return prompt

def send_to_openai_chat(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = None) -> str:
    """
    Send a prompt to OpenAI's Chat API and return the response.
    Uses secure API key handling and ensures privacy compliance.
    
    Args:
        prompt: Formatted prompt for GPT
        model: GPT model to use
        max_tokens: Maximum tokens for response
        
    Returns:
        GPT-generated response
    """
    # Get API key using secure method
    api_key = get_openai_api_key()
    if not api_key:
        return "Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable or configure it in the system keyring."
        
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Define messages for the conversation
    messages = [
        {"role": "system", "content": "You are a test automation expert who analyzes logs and provides clear, concise explanations of test failures."},
        {"role": "user", "content": prompt}
    ]
    
    # Use provided max_tokens or default
    if max_tokens is None:
        max_tokens = MAX_TOKENS
    
    # Create the payload without the metadata field
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE
    }
    
    # Add privacy-related headers instead of metadata
    headers["OpenAI-Beta"] = "optout=train"  # Optional: Signal not to use for training
    
    # Log the token counts
    prompt_tokens = get_accurate_token_count(prompt, model)
    logging.info(f"Prompt token count: {prompt_tokens}, Max response tokens: {max_tokens}")
    
    try:
        logging.info("Sending request to OpenAI API")
        
        # Add debug logging to help diagnose issues
        logging.debug(f"OpenAI API URL: {url}")
        logging.debug(f"OpenAI API model: {model}")
        logging.debug(f"OpenAI API payload size: {len(str(payload))} characters")
        
        response = requests.post(url, headers=headers, json=payload)
        
        # If we get an error, try to extract more details
        if not response.ok:
            error_details = "No error details available"
            try:
                error_json = response.json()
                if 'error' in error_json:
                    error_details = error_json['error'].get('message', str(error_json))
            except:
                error_details = response.text[:500] if response.text else "No response text"
                
            logging.error(f"API Error ({response.status_code}): {error_details}")
            return f"Error from OpenAI API ({response.status_code}): {error_details}\n\nPlease review the logs manually."
            
        result = response.json()
        logging.info("Received response from OpenAI API")
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling OpenAI API: {str(e)}"
        if hasattr(e, 'response') and e.response:
            error_msg += f" - Status Code: {e.response.status_code}"
            try:
                error_details = e.response.json()
                error_msg += f" - Details: {error_details}"
            except:
                pass
        logging.error(error_msg)
        return f"Error generating summary: {str(e)}\n\nPlease review the logs manually."

def fallback_summary(errors: List[Dict], clusters: Dict[int, List[Dict]], 
                    component_summary: List[Dict] = None, 
                    primary_issue_component: str = "unknown") -> str:
    """
    Generate a basic summary when GPT is not available.
    
    Args:
        errors: List of error dictionaries
        clusters: Dictionary mapping cluster IDs to lists of errors
        component_summary: Summary of components involved
        primary_issue_component: The component identified as root cause
        
    Returns:
        Basic summary of errors
    """
    high_severity = [e for e in errors if e.get('severity') == 'High']
    medium_severity = [e for e in errors if e.get('severity') == 'Medium']
    
    summary = f"Analysis based on {len(errors)} errors ({len(high_severity)} high severity, {len(medium_severity)} medium severity) grouped into {len(clusters)} clusters.\n\n"
    
    # Add component information
    if primary_issue_component != "unknown" and component_summary:
        primary_comp_info = next((c for c in component_summary if c["id"] == primary_issue_component), {})
        summary += f"PRIMARY ISSUE COMPONENT: {primary_comp_info.get('name', primary_issue_component.upper())}\n"
        summary += f"DESCRIPTION: {primary_comp_info.get('description', '')}\n"
        summary += f"ERROR COUNT: {primary_comp_info.get('error_count', 0)}\n\n"
    
    summary += "ROOT CAUSE:\n"
    if high_severity:
        error_text = high_severity[0].get('text', '')
        comp = high_severity[0].get('component', 'unknown').upper()
        summary += f"High severity error detected in {comp}: {error_text[:200]}\n"
    elif medium_severity:
        error_text = medium_severity[0].get('text', '')
        comp = medium_severity[0].get('component', 'unknown').upper()
        summary += f"Medium severity error detected in {comp}: {error_text[:200]}\n"
    else:
        summary += "No critical errors detected, but test failed.\n"
    
    summary += "\nIMPACT:\n"
    summary += f"Test failure with {len(errors)} errors.\n"
    
    summary += "\nRECOMMENDED ACTIONS:\n"
    if primary_issue_component != "unknown":
        summary += f"- Investigate issues in the {primary_issue_component.upper()} component.\n"
    summary += "- Check logs for more context on the failures.\n"
    summary += "- Review related components that might be affected.\n"
    
    return summary

def enrich_logs_with_errors(step_to_logs: Dict[int, List[Any]], errors: List[Dict]) -> Dict[int, List[Any]]:
    """
    Enrich log entries with error information.
    
    Args:
        step_to_logs: Dictionary mapping step numbers to log entries
        errors: List of error dictionaries
        
    Returns:
        Enriched step_to_logs dictionary
    """
    # Create lookup for errors by file and line
    error_lookup = {}
    for error in errors:
        key = (error.get('file', ''), error.get('line_num', 0))
        error_lookup[key] = error
    
    # Enrich log entries with error information
    for step_num, logs in step_to_logs.items():
        for log in logs:
            if hasattr(log, 'file') and hasattr(log, 'line_number'):
                key = (log.file, log.line_number)
                if key in error_lookup:
                    log.is_error = True
                    log.severity = error_lookup[key].get('severity', 'Low')
                    if 'component' in error_lookup[key]:
                        log.component = error_lookup[key]['component']
    
    return step_to_logs

def generate_summary_from_clusters(
    clusters: Dict[int, List[Dict]],
    ocr_data: List[Dict],
    test_id: str,
    scenario_text: str = "",
    use_gpt: bool = True,
    model: str = DEFAULT_MODEL,
    step_to_logs: Optional[Dict[int, List[Any]]] = None,
    feature_file: Optional[str] = None,
    component_analysis: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a summary from clustered errors using GPT with enhanced component analysis.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of errors
        ocr_data: OCR data extracted from images
        test_id: Test ID
        scenario_text: Feature file scenario text (optional)
        use_gpt: Whether to use GPT for summary generation
        model: GPT model to use
        step_to_logs: Dictionary mapping step numbers to log entries (optional)
        feature_file: Path to the feature file (optional)
        component_analysis: Results from component relationship analysis (optional)
        
    Returns:
        Generated summary
    """
    # Extract all errors from clusters
    all_errors = []
    for cluster_errs in clusters.values():
        all_errors.extend(cluster_errs)
    
    # If no errors or clusters found, provide a simple message
    if not all_errors or not clusters:
        return "No errors were found in the logs. Please check if test logs were properly collected."
    
    # Apply direct component mapping if component_analysis is not provided
    if component_analysis is None:
        try:
            logging.info("Applying direct component mapping for GPT summarization")
            errors_with_components, component_summary, primary_issue_component = assign_components_and_relationships(all_errors)
            all_errors = errors_with_components
            
            # Also apply component mapping to clustered errors
            enhanced_clusters = {}
            for cluster_id, cluster_errors in clusters.items():
                cluster_errors_with_components, _, _ = assign_components_and_relationships(cluster_errors)
                enhanced_clusters[cluster_id] = cluster_errors_with_components
            clusters = enhanced_clusters
            
            # Create basic component analysis
            component_analysis = {
                "primary_issue_component": primary_issue_component,
                "component_summary": component_summary
            }
            
            logging.info(f"Direct component analysis identified {primary_issue_component} as the primary issue component")
        except Exception as e:
            logging.error(f"Error applying direct component mapping: {str(e)}")
            primary_issue_component = "unknown"
            component_summary = []
            component_analysis = None
    
    # If not using GPT, generate a basic summary
    if not use_gpt or not get_openai_api_key():
        if not use_gpt:
            logging.info("Generating offline summary (GPT disabled)")
        else:
            logging.warning("No OpenAI API key found. Generating offline summary.")
        return fallback_summary(all_errors, clusters, component_summary if 'component_summary' in locals() else None, 
                              primary_issue_component if 'primary_issue_component' in locals() else "unknown")
    
    # Prepare data for token-managed prompt
    all_data = {
        "errors": all_errors,
        "clusters": clusters,
        "ocr_data": ocr_data,
        "scenario_text": scenario_text,
        "component_analysis": component_analysis
    }
    
    # Build token-managed prompt
    prompt = build_token_managed_prompt(test_id, all_data, model)
    
    # Send to OpenAI
    logging.info(f"Sending prompt to OpenAI API using model {model}")
    
    # Get model token limit
    model_limit = get_model_token_limits(model)
    
    # Calculate max_tokens (25% of limit)
    max_tokens = int(model_limit * 0.25)
    
    return send_to_openai_chat(prompt, model, max_tokens=max_tokens)

def test_token_management(test_id="SXM-3690790"):
    """
    Test the token management implementation.
    
    Args:
        test_id: Test ID to test with
        
    Returns:
        Success status
    """
    try:
        # Set up logging
        Config.setup_logging()
        
        # Import necessary modules
        from log_analyzer import parse_logs
        from log_segmenter import collect_log_files
        from error_clusterer import perform_error_clustering
        
        # Get logs
        log_path = os.path.join("logs", test_id)
        logs = collect_log_files(log_path)
        
        if not logs:
            logging.error(f"No logs found for test {test_id}")
            return False
            
        # Parse logs
        errors = parse_logs(logs)
        
        if not errors:
            logging.error(f"No errors found in logs for test {test_id}")
            return False
            
        # Cluster errors
        clusters = perform_error_clustering(errors)
        
        # Generate summary
        summary = generate_summary_from_clusters(
            clusters=clusters,
            ocr_data=[],
            test_id=test_id,
            model="gpt-3.5-turbo"
        )
        
        # Check if we got a response
        if "Error from OpenAI API" in summary:
            logging.error("Test failed: API error in response")
            return False
            
        logging.info("Test successful: Generated summary without token errors")
        return True
        
    except Exception as e:
        logging.error(f"Test failed with exception: {str(e)}")
        traceback.print_exc()
        return False