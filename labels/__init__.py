"""
Label mappings: point clouds keep fine component labels, a mapping turns them into the classes a task needs.

A mapping file (e.g. obm_mapping.json) has
    component_labels  component name -> id (the fine labels stored in the point clouds)
    systems           component name -> system name (coarse classes); a component that is itself a system maps to itself
    system_labels     system name -> id
    synthetic         source name (design / IFC model) -> component name
    physical          source name (annotation of real data) -> component name

    mapping = LabelMapping.load()                       # obm_mapping.json
    system_ids = mapping.to_systems(component_ids)      # numpy lookup, any shape
"""
import json
from pathlib import Path

import numpy as np

MAPPING_FOLDER = Path(__file__).parent


class LabelMapping:

    def __init__(self, mapping: dict, name=""):
        self.name = name
        self.metadata = mapping.get("metadata", {})
        self.component_labels = mapping["component_labels"]
        self.systems = mapping.get("systems", {})
        self.system_labels = mapping.get("system_labels", {})
        self.sources = {key: mapping[key] for key in ("synthetic", "physical") if key in mapping}
        ids = list(self.component_labels.values())
        if len(set(ids)) != len(ids):
            raise ValueError(f"Mapping {name}: component ids are not unique")

    @classmethod
    def load(cls, path="obm_mapping.json"):
        """A mapping file: a path, or the name of a file in railtwin_common/labels."""
        path = Path(path)
        if not path.exists():
            path = MAPPING_FOLDER / path
        with open(path, encoding="utf-8") as file:
            return cls(json.load(file), name=path.stem)

    def system_of(self, component):
        """System name of a component name (the component itself if it is not in 'systems')."""
        return self.systems.get(component, component)

    def lookup_table(self, target_labels: dict, target_of, default=None):
        """
        int32 table: table[component id] = target id.
        :param target_labels: target name -> id
        :param target_of: function component name -> target name
        :param default: target name for components without a target (None: raise and list them)
        """
        size = max(self.component_labels.values()) + 1
        fill = -1 if default is None else target_labels[default]
        table = np.full(size, fill, dtype=np.int32)
        missing = []
        for component, component_id in self.component_labels.items():
            target = target_of(component)
            if target in target_labels:
                table[component_id] = target_labels[target]
            elif default is None:
                missing.append(f"{component} -> {target}")
        if missing:
            raise KeyError(f"Mapping {self.name}: no target label for {', '.join(missing)}")
        return table

    def to_systems(self, component_ids, default=None):
        """System ids for an array of component ids (default: system name for components without one)."""
        table = self.lookup_table(self.system_labels, self.system_of, default)
        component_ids = np.asarray(component_ids)
        if component_ids.size and (component_ids.min() < 0 or component_ids.max() >= len(table)):
            raise ValueError(f"Component ids outside 0..{len(table) - 1} of mapping {self.name}")
        return table[component_ids]

    def component_id(self, source_name, source="physical"):
        """Component id of a name from a source ('synthetic' or 'physical')."""
        return self.component_labels[self.sources[source][source_name]]
