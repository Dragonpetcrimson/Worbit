# gherkin_log_correlator.py
import os
import re
import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# Try to import Config, but use a dummy if not available
try:
    from config import Config
    ENABLE_DIAGNOSTIC_CHECKS = getattr(Config, "ENABLE_DIAGNOSTIC_CHECKS", False)
except ImportError:
    logging.warning("Config module not available in gherkin_log_correlator.py, using defaults")
    class DummyConfig:
        ENABLE_DIAGNOSTIC_CHECKS = False
    Config = DummyConfig

# Log successful import if diagnostic checks are enabled
if getattr(Config, "ENABLE_DIAGNOSTIC_CHECKS", False):
    logging.info("Successfully imported Gherkin log correlator functions")

@dataclass
class GherkinStep:
    """Represents a single step in a Gherkin feature file."""
    keyword: str       # Given, When, Then, And, But, *
    text: str          # The step text without the keyword
    original_line: str # The full original line
    line_number: int   # Line number in the feature file
    step_number: int   # Sequential step number in the scenario
    scenario_name: str # Name of the parent scenario
    
    @property
    def full_text(self) -> str:
        """Returns the complete step text including keyword."""
        return f"{self.keyword} {self.text}"
    
    @property
    def normalized_text(self) -> str:
        """Returns text with common variables replaced with placeholders."""
        # Replace common patterns with placeholders
        normalized = re.sub(r'"([^"]*)"', '"PARAM"', self.text)
        normalized = re.sub(r'\d+', 'NUM', normalized)
        return normalized
    
    def extract_keywords(self) -> List[str]:
        """Extract important keywords from the step text."""
        # Remove common words, determiners, etc.
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'is', 'are', 'to', 'for', 'with', 'by', 'of', 'on', 'at'}
        words = re.findall(r'\b\w+\b', self.text.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]

class LogEntry:
    """Represents a parsed log entry with metadata."""
    def __init__(self, text: str, file: str, line_number: int, timestamp: Optional[datetime.datetime] = None):
        self.text = text
        self.file = file
        self.line_number = line_number
        self.timestamp = timestamp
        self.step_relevance_scores = {}  # Map of step_number -> relevance score
        self.assigned_step = None  # The step this entry is assigned to
        
    def __repr__(self) -> str:
        ts = f" @ {self.timestamp.isoformat()}" if self.timestamp else ""
        return f"LogEntry({self.file}:{self.line_number}{ts})"

class LogFormatAdapter:
    """Base class for log format adapters."""
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.format_name = "generic"
    
    @classmethod
    def can_handle(cls, log_file_path: str) -> bool:
        """Check if this adapter can handle the given log file."""
        return True  # Default adapter handles everything
    
    def extract_entries(self) -> List[LogEntry]:
        """Extract structured entries from this log file."""
        entries = []
        try:
            with open(self.log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f, 1):
                    timestamp = self._extract_timestamp(line)
                    entries.append(LogEntry(
                        text=line.strip(), 
                        file=os.path.basename(self.log_file_path),
                        line_number=i,
                        timestamp=timestamp
                    ))
        except Exception as e:
            logging.error(f"Error processing {self.log_file_path}: {e}")
        return entries
    
    def identify_step_transitions(self, entries: List[LogEntry]) -> List[LogEntry]:
        """Identify entries that likely indicate test step transitions."""
        # Default implementation returns no transition indicators
        return []
    
    def _extract_timestamp(self, line: str) -> Optional[datetime.datetime]:
        """Extract timestamp from a log line using various formats."""
        timestamp_patterns = [
            # ISO format: 2023-01-01T12:34:56.789Z or 2023-01-01 12:34:56,789
            (r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.,]\d+)', 
             lambda x: datetime.datetime.strptime(x.replace('T', ' ').replace(',', '.'), 
                                                '%Y-%m-%d %H:%M:%S.%f')),
            # Simple date time: 2023-01-01 12:34:56
            (r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', 
             lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')),
            # Time only: 12:34:56.789
            (r'(\d{2}:\d{2}:\d{2}[.,]\d+)', 
             lambda x: datetime.datetime.strptime(x.replace(',', '.'), '%H:%M:%S.%f').replace(
                 year=datetime.datetime.now().year,
                 month=datetime.datetime.now().month,
                 day=datetime.datetime.now().day)),
            # Simple time: 12:34:56
            (r'(\d{2}:\d{2}:\d{2})', 
             lambda x: datetime.datetime.strptime(x, '%H:%M:%S').replace(
                 year=datetime.datetime.now().year,
                 month=datetime.datetime.now().month,
                 day=datetime.datetime.now().day)),
        ]
        
        for pattern, parser in timestamp_patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    return parser(match.group(1))
                except ValueError:
                    continue
        return None

class AppiumLogAdapter(LogFormatAdapter):
    """Adapter for Appium logs."""
    def __init__(self, log_file_path: str):
        super().__init__(log_file_path)
        self.format_name = "appium"
    
    @classmethod
    def can_handle(cls, log_file_path: str) -> bool:
        if not log_file_path.endswith('.log'):
            return False
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                first_chunk = f.read(1000).lower()
                return 'appium' in first_chunk or 'webdriver' in first_chunk
        except:
            return False
    
    def identify_step_transitions(self, entries: List[LogEntry]) -> List[LogEntry]:
        transition_indicators = []
        for entry in entries:
            text_lower = entry.text.lower()
            # Look for command indicators in Appium logs
            if any(indicator in text_lower for indicator in [
                'find element', 
                'click element', 
                'send keys', 
                'new command', 
                'execute script'
            ]):
                transition_indicators.append(entry)
        return transition_indicators

class MimosaLogAdapter(LogFormatAdapter):
    """Adapter for Mimosa logs."""
    def __init__(self, log_file_path: str):
        super().__init__(log_file_path)
        self.format_name = "mimosa"
    
    @classmethod
    def can_handle(cls, log_file_path: str) -> bool:
        filename = os.path.basename(log_file_path).lower()
        if 'mimosa' in filename:
            return True
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                first_chunk = f.read(1000).lower()
                return 'mimosa' in first_chunk or 'signal' in first_chunk
        except:
            return False
    
    def identify_step_transitions(self, entries: List[LogEntry]) -> List[LogEntry]:
        transition_indicators = []
        for entry in entries:
            text_lower = entry.text.lower()
            # Look for command indicators in Mimosa logs
            if any(indicator in text_lower for indicator in [
                'request', 
                'response', 
                'signal', 
                'error',
                'started', 
                'completed'
            ]):
                transition_indicators.append(entry)
        return transition_indicators

class AppDebugLogAdapter(LogFormatAdapter):
    """Adapter for app_debug logs."""
    def __init__(self, log_file_path: str):
        super().__init__(log_file_path)
        self.format_name = "app_debug"
    
    @classmethod
    def can_handle(cls, log_file_path: str) -> bool:
        filename = os.path.basename(log_file_path).lower()
        if 'app_debug' in filename:
            return True
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                first_chunk = f.read(1000).lower()
                return 'androidx' in first_chunk or 'activity' in first_chunk
        except:
            return False
    
    def identify_step_transitions(self, entries: List[LogEntry]) -> List[LogEntry]:
        transition_indicators = []
        for entry in entries:
            text_lower = entry.text.lower()
            # Look for command indicators in app_debug logs
            if any(indicator in text_lower for indicator in [
                'activity', 
                'fragment', 
                'view', 
                'click',
                'started', 
                'created',
                'destroyed'
            ]):
                transition_indicators.append(entry)
        return transition_indicators

class ChromeLogAdapter(LogFormatAdapter):
    """Adapter for Chrome browser logs."""
    def __init__(self, log_file_path: str):
        super().__init__(log_file_path)
        self.format_name = "chrome"
    
    @classmethod
    def can_handle(cls, log_file_path: str) -> bool:
        if not log_file_path.endswith('.chlsj'):
            return False
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                first_chunk = f.read(100).lower()
                return '{' in first_chunk and 'webview' in first_chunk
        except:
            return False
    
    def extract_entries(self) -> List[LogEntry]:
        import json
        entries = []
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                try:
                    # Try parsing as JSON array
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, entry in enumerate(data):
                            timestamp = None
                            if 'timestamp' in entry:
                                try:
                                    timestamp_ms = int(entry['timestamp'])
                                    timestamp = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
                                except:
                                    pass
                            
                            message = entry.get('message', '')
                            entries.append(LogEntry(
                                text=message,
                                file=os.path.basename(self.log_file_path),
                                line_number=i+1,
                                timestamp=timestamp
                            ))
                except json.JSONDecodeError:
                    # Fall back to line-by-line reading
                    f.seek(0)
                    for i, line in enumerate(f, 1):
                        entries.append(LogEntry(
                            text=line.strip(),
                            file=os.path.basename(self.log_file_path),
                            line_number=i,
                            timestamp=self._extract_timestamp(line)
                        ))
        except Exception as e:
            logging.error(f"Error processing Chrome log {self.log_file_path}: {e}")
            
        return entries
    
    def identify_step_transitions(self, entries: List[LogEntry]) -> List[LogEntry]:
        transition_indicators = []
        for entry in entries:
            text_lower = entry.text.lower()
            if any(indicator in text_lower for indicator in [
                'navigated to', 
                'document.location', 
                'click', 
                'input', 
                'xhr'
            ]):
                transition_indicators.append(entry)
        return transition_indicators

def get_log_adapter(log_file_path: str) -> LogFormatAdapter:
    """Factory function to get the appropriate log adapter for a file."""
    adapters = [AppiumLogAdapter, ChromeLogAdapter, MimosaLogAdapter, AppDebugLogAdapter]
    
    for adapter_class in adapters:
        if adapter_class.can_handle(log_file_path):
            return adapter_class(log_file_path)
    
    # Default fallback
    return LogFormatAdapter(log_file_path)

class GherkinParser:
    """Parser for Gherkin feature files."""
    
    def __init__(self, feature_file_path: str):
        self.feature_file_path = feature_file_path
        self.steps = []
        self.background_steps = []
        self.scenarios = []
        
    def parse(self) -> List[GherkinStep]:
        """Parse the feature file and extract all steps."""
        if not os.path.exists(self.feature_file_path):
            logging.error(f"Feature file not found: {self.feature_file_path}")
            return []
            
        with open(self.feature_file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        in_background = False
        in_scenario = False
        current_scenario = ""
        step_count = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
                
            lower = stripped.lower()
            
            if lower.startswith('background:'):
                in_background = True
                in_scenario = False
                current_scenario = "Background"
                continue
                
            if lower.startswith(('scenario:', 'scenario outline:')):
                in_background = False
                in_scenario = True
                current_scenario = stripped.split(':', 1)[1].strip()
                self.scenarios.append(current_scenario)
                continue
                
            if (in_background or in_scenario) and re.match(r'^(given|when|then|and|but|\*)\s+', lower, re.IGNORECASE):
                keyword, text = re.match(r'^(given|when|then|and|but|\*)\s+(.*)', stripped, re.IGNORECASE).groups()
                step_count += 1
                step = GherkinStep(
                    keyword=keyword.capitalize(),
                    text=text,
                    original_line=stripped,
                    line_number=i+1,
                    step_number=step_count,
                    scenario_name=current_scenario
                )
                
                if in_background:
                    self.background_steps.append(step)
                else:
                    self.steps.append(step)
        
        # If we have background steps, prepend them to each scenario's steps
        if self.background_steps:
            # Adjust step numbers
            offset = len(self.background_steps)
            for step in self.steps:
                step.step_number += offset
                
            # Combine background and scenario steps
            return self.background_steps + self.steps
        else:
            return self.steps

class GherkinLogCorrelator:
    """Correlates Gherkin steps with log entries."""
    
    def __init__(self, feature_file_path: str, log_file_paths: List[str]):
        self.feature_file_path = feature_file_path
        self.log_file_paths = log_file_paths
        self.parser = GherkinParser(feature_file_path)
        self.steps = []
        self.log_entries = []
        self.step_to_logs = {}  # Map of step_number to list of log entries
        
    def analyze(self) -> Dict[int, List[LogEntry]]:
        """
        Analyze logs and correlate them with Gherkin steps.
        Returns a dict mapping step numbers to log entries.
        """
        # Parse Gherkin feature
        self.steps = self.parser.parse()
        if not self.steps:
            logging.error("No steps found in feature file")
            return {}
            
        # Parse logs
        self._parse_logs()
        if not self.log_entries:
            logging.error("No log entries found")
            return {}
            
        # Find step transitions in logs
        transitions = self._identify_step_transitions()
        
        # Do initial time-based assignment
        self._assign_by_timestamp(transitions)
        
        # Enhance with keyword matching
        self._enhance_with_keywords()
        
        # Return the correlated results
        # The step_to_logs dictionary is already keyed by step number
        return self.step_to_logs
        
    def _parse_logs(self) -> None:
        """Parse all log files and extract entries."""
        for log_path in self.log_file_paths:
            adapter = get_log_adapter(log_path)
            logging.info(f"Processing {log_path} with {adapter.format_name} adapter")
            entries = adapter.extract_entries()
            self.log_entries.extend(entries)
            
        # Sort all entries by timestamp if available
        self.log_entries.sort(
            key=lambda e: e.timestamp if e.timestamp else datetime.datetime.max
        )
            
    def _identify_step_transitions(self) -> List[Tuple[int, LogEntry]]:
        """
        Identify likely log entries that indicate transitions between steps.
        Returns a list of (log_index, entry) pairs.
        """
        transitions = []
        
        # Group logs by file
        logs_by_file = {}
        for i, entry in enumerate(self.log_entries):
            if entry.file not in logs_by_file:
                logs_by_file[entry.file] = []
            logs_by_file[entry.file].append((i, entry))
            
        # Process each file with its appropriate adapter
        for file, entries in logs_by_file.items():
            log_path = next((p for p in self.log_file_paths if os.path.basename(p) == file), None)
            if not log_path:
                continue
                
            adapter = get_log_adapter(log_path)
            # Extract just the LogEntry objects
            file_entries = [e for _, e in entries]
            # Get transition indicators
            indicators = adapter.identify_step_transitions(file_entries)
            
            # Map back to global indices
            for indicator in indicators:
                for idx, entry in entries:
                    if entry is indicator:
                        transitions.append((idx, entry))
                        break
                        
        return sorted(transitions, key=lambda x: x[0])
        
    def _assign_by_timestamp(self, transitions: List[Tuple[int, LogEntry]]) -> None:
        """
        Assign log entries to steps based on timestamps and transitions.
        """
        # Initialize step_to_logs - use step numbers as keys
        self.step_to_logs = {step.step_number: [] for step in self.steps}
        
        if not transitions:
            # If no transitions detected, split logs evenly among steps
            chunk_size = len(self.log_entries) // len(self.steps)
            for i, step in enumerate(self.steps):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < len(self.steps) - 1 else len(self.log_entries)
                self.step_to_logs[step.step_number] = self.log_entries[start_idx:end_idx]
                # Mark assigned step
                for entry in self.log_entries[start_idx:end_idx]:
                    entry.assigned_step = step.step_number
        else:
            # If we have transitions, use them as step boundaries
            step_idx = 0
            current_step = self.steps[step_idx].step_number if self.steps else 1
            
            # Process logs before first transition
            first_transition_idx = transitions[0][0]
            for i in range(first_transition_idx):
                self.log_entries[i].assigned_step = current_step
                self.step_to_logs[current_step].append(self.log_entries[i])
                
            # Process logs between transitions
            for i in range(len(transitions) - 1):
                current_idx, _ = transitions[i]
                next_idx, _ = transitions[i + 1]
                
                # Move to next step if we have steps left
                step_idx = min(step_idx + 1, len(self.steps) - 1)
                current_step = self.steps[step_idx].step_number
                
                # Assign logs to current step
                for j in range(current_idx, next_idx):
                    self.log_entries[j].assigned_step = current_step
                    self.step_to_logs[current_step].append(self.log_entries[j])
                    
            # Process logs after last transition
            last_idx = transitions[-1][0]
            step_idx = min(step_idx + 1, len(self.steps) - 1)
            current_step = self.steps[step_idx].step_number
            
            for i in range(last_idx, len(self.log_entries)):
                self.log_entries[i].assigned_step = current_step
                self.step_to_logs[current_step].append(self.log_entries[i])
    
    def _enhance_with_keywords(self) -> None:
        """
        Enhance step assignment using keyword matching between steps and logs.
        """
        # Calculate relevance scores
        for entry in self.log_entries:
            for step in self.steps:
                keywords = step.extract_keywords()
                if not keywords:
                    continue
                    
                matches = 0
                for keyword in keywords:
                    if keyword.lower() in entry.text.lower():
                        matches += 1
                        
                score = matches / len(keywords) if keywords else 0
                entry.step_relevance_scores[step.step_number] = score
                
        # Reassign entries with strong keyword matches to other steps
        reassignments = []
        
        for step_num, logs in self.step_to_logs.items():
            for entry in logs:
                current_score = entry.step_relevance_scores.get(step_num, 0)
                
                # Check if entry has stronger match with another step
                for other_step in self.steps:
                    if other_step.step_number == step_num:
                        continue
                        
                    other_score = entry.step_relevance_scores.get(other_step.step_number, 0)
                    # If score difference is significant, plan reassignment
                    if other_score > 0.5 and other_score > current_score + 0.3:
                        reassignments.append((entry, step_num, other_step.step_number))
                        break
                        
        # Apply reassignments
        for entry, from_step, to_step in reassignments:
            if entry in self.step_to_logs[from_step]:
                self.step_to_logs[from_step].remove(entry)
                self.step_to_logs[to_step].append(entry)
                entry.assigned_step = to_step

def correlate_logs_with_steps(feature_file_path: str, log_file_paths: List[str]) -> Dict[int, List[LogEntry]]:
    """
    Correlate log entries with Gherkin steps.
    
    Args:
        feature_file_path: Path to the Gherkin feature file
        log_file_paths: List of paths to log files
        
    Returns:
        Dictionary mapping step numbers to lists of log entries
    """
    # Additional logging for diagnostics
    if getattr(Config, "ENABLE_DIAGNOSTIC_CHECKS", False):
        logging.info(f"Correlating logs for test with feature file: {feature_file_path}")
        logging.info(f"Log files to correlate: {len(log_file_paths)}")
        for log_file in log_file_paths:
            logging.info(f"  - {log_file}")
    
    correlator = GherkinLogCorrelator(feature_file_path, log_file_paths)
    return correlator.analyze()

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python gherkin_log_correlator.py <feature_file> <log_file1> [log_file2 ...]")
        sys.exit(1)
        
    feature_file = sys.argv[1]
    log_files = sys.argv[2:]
    
    logging.basicConfig(level=logging.INFO)
    results = correlate_logs_with_steps(feature_file, log_files)
    
    # Print summary
    print(f"\nFound {len(results)} steps with correlated logs:")
    for step_num, logs in sorted(results.items()):
        print(f"Step {step_num}: {len(logs)} log entries")