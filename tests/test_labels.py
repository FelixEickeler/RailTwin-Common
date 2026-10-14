import numpy as np
import pytest

from railtwin_common.labels import LabelMapping


@pytest.fixture
def obm():
    return LabelMapping.load()


def test_obm_mapping_loads(obm):
    assert obm.name == "obm_mapping"
    assert len(obm.component_labels) == 64
    assert obm.component_labels["duct"] == 63


def test_to_systems(obm):
    c = obm.component_labels
    ids = np.array([[c["contact-wire"], c["dropper"]], [c["rail"], c["weed"]]])
    s = obm.system_labels
    np.testing.assert_array_equal(obm.to_systems(ids, default="instance"),
                                  [[s["wire"], s["wire"]], [s["rail"], s["vegetation"]]])


def test_components_without_system_are_listed(obm):
    with pytest.raises(KeyError, match="bridge"):
        obm.to_systems([0])
    assert obm.to_systems([obm.component_labels["bridge"]], default="instance")[0] == obm.system_labels["instance"]


def test_ids_out_of_range(obm):
    with pytest.raises(ValueError):
        obm.to_systems([999], default="instance")


def test_source_names(obm):
    assert obm.component_id("Schwelle", "synthetic") == obm.component_labels["sleeper"]
    assert obm.component_id("catenary") == obm.component_labels["catenary-wire"]


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        LabelMapping({"component_labels": {"a": 1, "b": 1}})
