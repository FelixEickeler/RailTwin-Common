"""
Label mappings: point clouds keep fine component labels, a mapping turns them into the classes a task needs.

A mapping file (e.g. obm_mapping.json) has
    component_labels  component name -> id (the fine labels stored in the point clouds)
    systems           component name -> system name; a component that is itself a system maps to itself
    system_labels     system name -> id
    synthetic         source name (design / IFC model) -> component name
    physical          source name (annotation of real data) -> component name
    levels            optional, more class sets:
                      {"<level>": {"labels": {class: id},          classes of the level
                                   "components": {component: class}, components not listed map to the class of
                                                                     the same name, if there is one
                                   "default": class,                 class of all other components (optional)
                                   "ignored": [class, ...]}}         classes not trained (optional)

Every mapping has the levels "component" (the components themselves) and "system" (systems / system_labels).

    mapping = LabelMapping.load()                       # labels/obm_mapping.json, or any path
    level = mapping.level("system")                     # level.labels, level.ignored, level.names
    ids = mapping.remap(component_ids, "system")        # numpy lookup, any shape

Mappings can be chained: the classes of one step are (by name) the components of the next step, e.g. the labels of
an input dataset -> consolidated components -> training classes.

    chain = LabelChain.load([["input_mapping.json", "consolidated"], ["obm_mapping.json", "system"]])
    chain = LabelChain.load("obm_mapping.json", "system")   # a single mapping is a chain of one step
    ids = chain.remap(input_ids)                            # chain.level: the classes at the end
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAPPING_FOLDER = Path(__file__).parent


@dataclass
class LabelLevel:
    """One class set: its labels (name -> id), the class of each component, a default and the ignored classes."""
    name: str
    labels: dict
    components: dict = field(default_factory=dict)
    default: str = None
    ignored: list = field(default_factory=list)

    def class_of(self, component):
        """Class name of a component name (None if it has none)."""
        target = self.components.get(component, component)
        if target in self.labels:
            return target
        return self.default

    @property
    def names(self):
        """id -> class name"""
        return {i: name for name, i in self.labels.items()}

    @property
    def ignored_ids(self):
        return [self.labels[name] for name in self.ignored]


class LabelMapping:

    def __init__(self, mapping: dict, name=""):
        self.name = name
        self.metadata = mapping.get("metadata", {})
        self.component_labels = mapping["component_labels"]
        self.sources = {key: mapping[key] for key in ("synthetic", "physical") if key in mapping}
        _check_unique(self.component_labels, f"{name} component_labels")

        self.levels = {"component": LabelLevel("component", dict(self.component_labels),
                                               ignored=[c for c in ("unclassified",) if c in self.component_labels])}
        if "system_labels" in mapping:
            system_labels = mapping["system_labels"]
            # Components without a system go to "instance" (unclassified / ignored things), if the mapping has it
            default = "instance" if "instance" in system_labels else None
            self.levels["system"] = LabelLevel("system", dict(system_labels), dict(mapping.get("systems", {})),
                                               default=default, ignored=[default] if default else [])
        for level_name, level in mapping.get("levels", {}).items():
            self.levels[level_name] = LabelLevel(level_name, dict(level["labels"]), dict(level.get("components", {})),
                                                 level.get("default"), list(level.get("ignored", [])))
        for level in self.levels.values():
            _check_unique(level.labels, f"{name} level {level.name}")
            unknown = [c for c in level.ignored + ([level.default] if level.default else []) if c not in level.labels]
            if unknown:
                raise ValueError(f"Mapping {name}, level {level.name}: unknown classes {unknown}")

    @classmethod
    def load(cls, path="obm_mapping.json"):
        """A mapping file: a path, or the name of a file in railtwin_common/labels."""
        path = Path(path)
        if not path.exists():
            path = MAPPING_FOLDER / path
        with open(path, encoding="utf-8") as file:
            return cls(json.load(file), name=path.stem)

    def level(self, name) -> LabelLevel:
        if name not in self.levels:
            raise KeyError(f"Mapping {self.name} has no level '{name}' (levels: {', '.join(self.levels)})")
        return self.levels[name]

    def unmapped(self, level):
        """Components that have no class in a level and no default."""
        level = self.level(level) if isinstance(level, str) else level
        return [c for c in self.component_labels if level.class_of(c) is None]

    def lookup_table(self, level):
        """int32 table: table[component id] = class id of the level. Raises if a component has no class."""
        level = self.level(level) if isinstance(level, str) else level
        missing = self.unmapped(level)
        if missing:
            raise KeyError(f"Mapping {self.name}, level {level.name}: no class for {', '.join(missing)} "
                           f"(add them to the level or give it a default)")
        table = np.zeros(max(self.component_labels.values()) + 1, dtype=np.int32)
        for component, component_id in self.component_labels.items():
            table[component_id] = level.labels[level.class_of(component)]
        return table

    def remap(self, component_ids, level="system"):
        """Class ids of a level for an array of component ids (any shape)."""
        table = self.lookup_table(level)
        component_ids = np.asarray(component_ids).astype(np.int64)
        known = np.isin(component_ids, list(self.component_labels.values()))
        if not known.all():
            raise ValueError(f"Component ids {np.unique(component_ids[~known])[:10].tolist()} are not in "
                             f"mapping {self.name}")
        return table[component_ids]

    def to_systems(self, component_ids):
        return self.remap(component_ids, "system")

    def component_id(self, source_name, source="physical"):
        """Component id of a name from a source ('synthetic' or 'physical')."""
        return self.component_labels[self.sources[source][source_name]]


class LabelChain:
    """Label mappings applied one after another; the classes of a step are the components of the next, by name."""

    def __init__(self, steps):
        """:param steps: [(LabelMapping, level name), ...]"""
        if not steps:
            raise ValueError("A label chain needs at least one step")
        self.steps = [(mapping, mapping.level(level)) for mapping, level in steps]
        for (mapping, level), (next_mapping, _) in zip(self.steps, self.steps[1:]):
            missing = [name for name in level.labels if name not in next_mapping.component_labels]
            if missing:
                raise KeyError(f"Classes {missing} of {mapping.name} level {level.name} are not components of "
                               f"{next_mapping.name}")
        self.name = "+".join(f"{mapping.name}_{level.name}" for mapping, level in self.steps)
        self.component_labels = self.steps[0][0].component_labels   # labels of the input data
        self.level = self.steps[-1][1]                              # classes at the end of the chain

    @classmethod
    def load(cls, steps, level="system"):
        """steps: a mapping file (with level), or a list of [mapping file, level] pairs."""
        if isinstance(steps, (str, Path)):
            steps = [(steps, level)]
        return cls([(LabelMapping.load(path), step_level) for path, step_level in steps])

    def lookup_table(self):
        """int32 table: table[input id] = class id at the end of the chain."""
        mapping, level = self.steps[0]
        table = mapping.lookup_table(level)
        for next_mapping, next_level in self.steps[1:]:
            # class ids of this step -> component ids of the next step
            to_next = np.zeros(max(level.labels.values()) + 1, dtype=np.int32)
            for name, class_id in level.labels.items():
                to_next[class_id] = next_mapping.component_labels[name]
            table = next_mapping.lookup_table(next_level)[to_next[table]]
            mapping, level = next_mapping, next_level
        return table

    def remap(self, input_ids):
        """Class ids at the end of the chain for an array of input ids (any shape)."""
        table = self.lookup_table()
        input_ids = np.asarray(input_ids).astype(np.int64)
        known = np.isin(input_ids, list(self.component_labels.values()))
        if not known.all():
            raise ValueError(f"Ids {np.unique(input_ids[~known])[:10].tolist()} are not in {self.steps[0][0].name}")
        return table[input_ids]


def _check_unique(labels, what):
    ids = list(labels.values())
    if len(set(ids)) != len(ids):
        raise ValueError(f"{what}: ids are not unique")
