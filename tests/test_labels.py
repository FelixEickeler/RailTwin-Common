import json

import numpy as np
import pytest

from railtwin_common.labels import LabelChain, LabelMapping


@pytest.fixture
def obm():
    return LabelMapping.load()


def test_obm_mapping_loads(obm):
    assert obm.name == "obm_mapping"
    assert len(obm.component_labels) == 64
    assert obm.component_labels["duct"] == 63
    assert set(obm.levels) == {"component", "system"}


def test_system_level(obm):
    c, s = obm.component_labels, obm.level("system").labels
    ids = np.array([[c["contact-wire"], c["dropper"]], [c["rail"], c["weed"]]])
    np.testing.assert_array_equal(obm.remap(ids, "system"), [[s["wire"], s["wire"]], [s["rail"], s["vegetation"]]])
    # Components without a system go to "instance", which is ignored
    assert obm.to_systems([c["bridge"]])[0] == s["instance"]
    assert obm.level("system").ignored == ["instance"]


def test_component_level_is_the_identity(obm):
    ids = np.array(sorted(obm.component_labels.values()))
    np.testing.assert_array_equal(obm.remap(ids, "component"), ids)
    assert obm.level("component").ignored == ["unclassified"]


def test_unknown_ids_and_levels(obm):
    with pytest.raises(ValueError):
        obm.remap([999])
    with pytest.raises(KeyError, match="levels: component, system"):
        obm.level("nope")


def test_source_names(obm):
    assert obm.component_id("Schwelle", "synthetic") == obm.component_labels["sleeper"]
    assert obm.component_id("catenary") == obm.component_labels["catenary-wire"]


def test_custom_level_from_the_file(tmp_path):
    mapping = {"component_labels": {"unclassified": 0, "rail": 1, "sleeper": 2, "contact-wire": 3, "dropper": 4},
               "levels": {"track_wire": {"labels": {"ignored": 0, "track": 1, "wire": 2},
                                         "components": {"rail": "track", "sleeper": "track",
                                                        "contact-wire": "wire", "dropper": "wire"},
                                         "default": "ignored", "ignored": ["ignored"]}}}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(mapping))
    m = LabelMapping.load(path)
    np.testing.assert_array_equal(m.remap([0, 1, 2, 3, 4], "track_wire"), [0, 1, 1, 2, 2])
    level = m.level("track_wire")
    assert level.ignored_ids == [0] and level.names == {0: "ignored", 1: "track", 2: "wire"}


def test_level_without_default_lists_missing_components():
    m = LabelMapping({"component_labels": {"rail": 1, "pylon": 2},
                      "levels": {"x": {"labels": {"rail": 0}}}})
    assert m.unmapped("x") == ["pylon"]
    with pytest.raises(KeyError, match="pylon"):
        m.remap([1], "x")


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        LabelMapping({"component_labels": {"a": 1, "b": 1}})


def test_chain_input_to_consolidated_to_training(tmp_path, obm):
    # An input dataset with its own ids, consolidated onto the obm components, then trained on the systems
    input_mapping = {"component_labels": {"unlabelled": 0, "Fahrdraht": 1, "Hänger": 2, "Schiene": 3, "Gras": 4},
                     "levels": {"consolidated": {
                         "labels": {"unclassified": 0, "contact-wire": 47, "dropper": 50, "rail": 16, "weed": 7},
                         "components": {"unlabelled": "unclassified", "Fahrdraht": "contact-wire",
                                        "Hänger": "dropper", "Schiene": "rail", "Gras": "weed"}}}}
    path = tmp_path / "input_mapping.json"
    path.write_text(json.dumps(input_mapping, ensure_ascii=False), encoding="utf-8")
    chain = LabelChain.load([[path, "consolidated"], ["obm_mapping.json", "system"]])
    s = obm.level("system").labels
    np.testing.assert_array_equal(chain.remap([0, 1, 2, 3, 4]),
                                  [s["instance"], s["wire"], s["wire"], s["rail"], s["vegetation"]])
    assert chain.level.name == "system" and chain.name == "input_mapping_consolidated+obm_mapping_system"


def test_single_mapping_is_a_chain_of_one(obm):
    chain = LabelChain.load("obm_mapping.json", "system")
    ids = np.array(sorted(obm.component_labels.values()))
    np.testing.assert_array_equal(chain.remap(ids), obm.remap(ids, "system"))
    assert chain.name == "obm_mapping_system"


def test_chain_classes_must_be_next_components(obm):
    bad = LabelMapping({"component_labels": {"a": 0}, "levels": {"x": {"labels": {"no-such-component": 0},
                                                                       "default": "no-such-component"}}})
    with pytest.raises(KeyError, match="no-such-component"):
        LabelChain([(bad, "x"), (obm, "system")])
