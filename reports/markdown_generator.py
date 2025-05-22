"""
reports/markdown_generator.py - Markdown report generation
"""

import os
import logging
from typing import List, Dict, Any

from reports.base import ReportGenerator, ReportConfig, ReportData
from utils.path_utils import (
    get_output_path,
    OutputType,
    normalize_test_id,
    get_standardized_filename
)

class MarkdownReportGenerator(ReportGenerator):
    """Generator for Markdown reports."""
    
    def generate(self, data: ReportData) -> str:
        """
        Generate a Markdown report.
        
        Args:
            data: Report data
            
        Returns:
            Path to the generated report
        """
        # Create the Markdown content
        md_output = f"# Test Analysis for {self.config.test_id}\n\n"
        md_output += "## AI-Generated Summary\n\n"
        md_output += f"{data.summary}\n\n"
        
        # Add component information to markdown
        component_analysis = data.component_analysis
        primary_issue_component = self.config.primary_issue_component
        
        if component_analysis and component_analysis.get("metrics", {}).get("component_tagged_errors", 0) > 0:
            md_output += "## Component Analysis\n\n"
            md_output += f"* Analyzed {component_analysis['metrics']['component_tagged_logs']} log entries\n"
            md_output += f"* Found {component_analysis['metrics']['component_tagged_errors']} errors across components\n"
            md_output += f"* Primary issue component: {primary_issue_component.upper()}\n"
            
            # Add component report link if available
            component_report_path = component_analysis.get("report_path", "")
            if component_report_path:
                report_filename = os.path.basename(component_report_path)
                md_output += f"\nSee detailed [Component Analysis Report]({report_filename}) for component relationships and error propagation.\n\n"
        elif primary_issue_component != "unknown":
            # Add direct component analysis to markdown
            md_output += "## Component Analysis\n\n"
            md_output += f"* Primary issue component: {primary_issue_component.upper()}\n"
            
            # Find component description from component analysis or fallback info
            component_description = ""
            if component_analysis and "component_summary" in component_analysis:
                component_description = next((c.get('description', '') for c in component_analysis["component_summary"] 
                                           if c.get('id') == primary_issue_component), '')
            
            if not component_description:
                # Fallback to component_info
                component_description = component_analysis.get("component_info", {}).get(primary_issue_component, {}).get('description', '')
                
            md_output += f"* Component description: {component_description}\n"
            
            # Add error count
            error_count = 0
            if component_analysis and "component_error_counts" in component_analysis:
                error_count = component_analysis["component_error_counts"].get(primary_issue_component, 0)
            else:
                error_count = sum(1 for err in data.errors if isinstance(err, dict) and 
                                 err.get('component') == primary_issue_component)
            
            md_output += f"* Error count: {error_count}\n\n"
            
            # Add affected components
            affected_components = []
            if component_analysis and "component_summary" in component_analysis:
                component_info = component_analysis.get("component_info", {})
                for comp in component_analysis["component_summary"]:
                    if comp["id"] in component_info.get(primary_issue_component, {}).get('related_to', []) and comp.get('error_count', 0) > 0:
                        affected_components.append(comp)
            
            if affected_components:
                md_output += "### Affected Components\n\n"
                for comp in affected_components:
                    md_output += f"* {comp['name']} ({comp['error_count']} errors): {comp['description']}\n"
            
            md_output += "\n"
        
        md_output += "## Key Errors\n\n"
        high_severity_errors = [err for err in data.errors if isinstance(err, dict) and err.get('severity') == 'High']
        for i, err in enumerate(high_severity_errors[:5], 1):
            # Make sure err is a dictionary before accessing keys
            if not isinstance(err, dict):
                continue
                
            component_info = f" [Component: {err.get('component', 'unknown').upper()}]" if 'component' in err else ""
            error_text = err.get('text', 'No error text available')
            if isinstance(error_text, str):
                error_text = error_text[:200]  # Truncate long error text
            else:
                error_text = str(error_text)[:200]  # Convert non-string values to string
            
            md_output += f"{i}. **{err.get('file', 'Unknown file')}**{component_info}: {error_text}...\n\n"
        
        # Update path to use standardized filename
        filename = get_standardized_filename(self.config.test_id, "log_analysis", "md")
        output_path = get_output_path(
            self.config.output_dir,
            self.config.test_id,
            filename,
            OutputType.PRIMARY_REPORT
        )
        
        # Write the Markdown file
        with open(output_path, "w", encoding='utf-8') as md_file:
            md_file.write(md_output)
        
        logging.info(f"Markdown report written to: {output_path}")
        return output_path