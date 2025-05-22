"""
Component Model Module - components/component_model.py

This module provides the core component model for the Orbit Analyzer,
serving as the single source of truth for component definitions and relationships.
It implements consistent component identification, information storage, and relationship tracking.

Key Classes:
- ComponentInfo: Immutable container for component information
- ComponentRegistry: Registry for loading and managing components
"""

import os
import json
import copy
import logging
from typing import Dict, List, Optional, Set, Tuple, Any


class ComponentInfo:
    """
    Immutable container for component information.
    Serves as the single source of truth for component data.
    """
    
    def __init__(self, component_id, name=None, description=None, component_type=None, **kwargs):
        """Initialize with required component data."""
        self._id = component_id.lower() if component_id else "unknown"
        self._name = name or self._id.upper()
        self._description = description or ""
        self._type = component_type or "unknown"
        self._source = kwargs.get('component_source', 'default')
        self._parent = kwargs.get('parent', None)
        self._children = kwargs.get('children', [])
        self._related_to = kwargs.get('related_to', [])
        self._properties = {**kwargs}  # Store all additional properties
        
    @property
    def id(self):
        """Get component ID (immutable)."""
        return self._id
        
    @property
    def name(self):
        """Get component name (immutable)."""
        return self._name
    
    @property
    def description(self):
        """Get component description (immutable)."""
        return self._description
    
    @property
    def type(self):
        """Get component type (immutable)."""
        return self._type
    
    @property
    def source(self):
        """Get identification source (immutable)."""
        return self._source
    
    @property
    def parent(self):
        """Get parent component ID (immutable)."""
        return self._parent
    
    @property
    def children(self):
        """Get child component IDs (immutable)."""
        return self._children
    
    @property
    def related_to(self):
        """Get related component IDs (immutable)."""
        return self._related_to
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'component': self._id,
            'component_source': self._source,
            'component_name': self._name,
            'component_description': self._description,
            'component_type': self._type,
            'parent_component': self._parent,
            'child_components': self._children,
            'related_components': self._related_to,
            **{k: v for k, v in self._properties.items() if k not in ['component', 'component_source']}
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary representation."""
        if not data:
            return cls("unknown")
            
        component_id = data.get('component', 'unknown')
        source = data.get('component_source', 'default')
        
        return cls(
            component_id=component_id,
            name=data.get('component_name', component_id.upper() if component_id else "UNKNOWN"),
            description=data.get('component_description', ""),
            component_type=data.get('component_type', "unknown"),
            component_source=source,
            parent=data.get('parent_component', None),
            children=data.get('child_components', []),
            related_to=data.get('related_components', []),
            **{k: v for k, v in data.items() if k not in cls.RESERVED_FIELDS}
        )
        
    RESERVED_FIELDS = {
        'component', 'component_source', 'component_name', 
        'component_description', 'component_type', 'parent_component',
        'child_components', 'related_components'
    }


class ComponentRegistry:
    """
    Registry for component information.
    Manages loading from schema and provides access to components.
    """
    
    def __init__(self, schema_path=None):
        """Initialize registry with schema path."""
        self._components = {}
        self._load_schema(schema_path)
        
    def _load_schema(self, schema_path):
        """Load components from schema."""
        try:
            if schema_path and os.path.exists(schema_path):
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                    
                for component_data in schema.get('components', []):
                    component_id = component_data.get('id')
                    if component_id:
                        self._components[component_id] = ComponentInfo(
                            component_id=component_id,
                            name=component_data.get('name'),
                            description=component_data.get('description'),
                            component_type=component_data.get('type'),
                            component_source='schema',
                            parent=component_data.get('parent'),
                            children=[],  # Will be populated later
                            related_to=component_data.get('receives_from', []) + 
                                      component_data.get('sends_to', []),
                            log_sources=component_data.get('logSources', []),
                            error_patterns=component_data.get('errorPatterns', [])
                        )
                
                # Set up parent-child relationships
                for component_id, component in self._components.items():
                    parent_id = component._parent
                    if parent_id and parent_id in self._components:
                        # Add this component as a child of its parent
                        parent_component = self._components[parent_id]
                        if component_id not in parent_component._children:
                            # Need to modify the immutable children list
                            parent_component._children = parent_component._children + [component_id]
            
            # Add fallback components if needed
            self._ensure_default_components()
                
        except Exception as e:
            logging.error(f"Error loading component schema: {str(e)}")
            self._ensure_default_components()
    
    def _ensure_default_components(self):
        """Ensure minimum required components exist."""
        default_components = {
            'soa': ComponentInfo('soa', 'SOA', 'SiriusXM application built on Android', 'application'),
            'android': ComponentInfo('android', 'Android', 'Android platform', 'platform'),
            'mimosa': ComponentInfo('mimosa', 'Mimosa', 'Test data provider', 'test_data_provider'),
            'phoebe': ComponentInfo('phoebe', 'Phoebe', 'Data transmission proxy', 'proxy'),
            'charles': ComponentInfo('charles', 'Charles', 'Proxy for live data', 'proxy'),
            'unknown': ComponentInfo('unknown', 'Unknown', 'Unknown component', 'unknown')
        }
        
        # Add any missing default components
        for comp_id, comp_info in default_components.items():
            if comp_id not in self._components:
                self._components[comp_id] = comp_info
    
    def get_component(self, component_id):
        """Get component by ID."""
        if not component_id or not isinstance(component_id, str):
            return self._components.get('unknown')
            
        component_id = component_id.lower()
        return self._components.get(component_id, self._components.get('unknown'))
    
    def get_all_components(self):
        """Get all registered components."""
        return list(self._components.values())
        
    def identify_component_from_filename(self, filename):
        """Identify component based on filename patterns."""
        if not filename:
            return self.get_component('unknown')
            
        filename = filename.lower()
        
        # Special cases first
        if 'app_debug.log' in filename:
            return self.get_component('soa')
        elif '.har' in filename or '.chlsj' in filename:
            return self.get_component('ip_traffic')
            
        # Check pattern matches from schema
        for component_id, component in self._components.items():
            log_sources = getattr(component, '_properties', {}).get('log_sources', [])
            for pattern in log_sources:
                if self._matches_pattern(filename, pattern):
                    return self.get_component(component_id)
        
        # Fallback to base filename
        base_name = os.path.basename(filename)
        component_id = os.path.splitext(base_name)[0]
        return self.get_component(component_id)
    
    def _matches_pattern(self, text, pattern):
        """Check if text matches a pattern, handling wildcards."""
        if '*' not in pattern:
            return text == pattern
            
        parts = pattern.split('*')
        if len(parts) == 2:
            # Simple wildcard pattern (e.g., "prefix*" or "*suffix")
            if not parts[0]:  # "*suffix"
                return text.endswith(parts[1])
            elif not parts[1]:  # "prefix*"
                return text.startswith(parts[0])
            else:  # "prefix*suffix"
                return text.startswith(parts[0]) and text.endswith(parts[1])
        
        # More complex wildcard patterns
        import re
        regex = pattern.replace('.', '\\.').replace('*', '.*')
        return bool(re.match(f"^{regex}$", text))
        
    def identify_primary_component(self, component_counts):
        """Identify primary component based on error counts."""
        if not component_counts:
            return self.get_component('unknown')
            
        # Filter out unknown component for primary selection
        filtered_counts = {k: v for k, v in component_counts.items() 
                          if k != 'unknown'}
        
        # Apply special SOA vs Android logic
        if 'soa' in filtered_counts and 'android' in filtered_counts:
            # If SOA errors are at least 50% of Android errors, prefer SOA
            if filtered_counts['soa'] >= filtered_counts['android'] * 0.5:
                # Boost SOA weight for primary component selection
                filtered_counts['soa'] = filtered_counts['soa'] * 1.5
        
        # Select primary component - prefer non-unknown components
        if filtered_counts:
            primary_id = max(filtered_counts.items(), key=lambda x: x[1])[0]
            return self.get_component(primary_id)
            
        # Last resort - use the most frequent component
        if component_counts:
            primary_id = max(component_counts.items(), key=lambda x: x[1])[0]
            return self.get_component(primary_id)
        return self.get_component('unknown')


# Factory functions for component creation
def create_component_info(component_id, source='default', **kwargs):
    """Create a ComponentInfo object with standard fields."""
    registry = get_component_registry()
    base_component = registry.get_component(component_id)
    
    extra = {k: v for k, v in kwargs.items()
             if k not in {'name', 'description', 'component_type', 'component_source'}}

    return ComponentInfo(
        component_id=component_id,
        name=kwargs.get('name', base_component.name),
        description=kwargs.get('description', base_component.description),
        component_type=kwargs.get('component_type', base_component.type),
        component_source=source,
        **extra
    )

# Singleton registry instance
_COMPONENT_REGISTRY = None

def get_component_registry(schema_path=None):
    """Get or create the component registry singleton."""
    global _COMPONENT_REGISTRY
    if _COMPONENT_REGISTRY is None:
        _COMPONENT_REGISTRY = ComponentRegistry(schema_path)
    return _COMPONENT_REGISTRY
