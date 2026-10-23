"""DB row -> required metadata for the internal conversion driver."""

from types import SimpleNamespace

import pytest

run_conversion = pytest.importorskip("egomimic.scripts.run_conversion")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"pink cup","white saucer"}', ["pink cup", "white saucer"]),
        ("{bag,fruits}", ["bag", "fruits"]),
        ("{None}", []),
        ("{bag,NULL}", ["bag"]),
        ("cup, towel", ["cup", "towel"]),
        ("", []),
        (None, []),
        (float("nan"), []),
        (["cup", None, "bowl"], ["cup", "bowl"]),
        ('{"say \\"hi\\"",cup}', ['say "hi"', "cup"]),
    ],
)
def test_parse_db_objects(raw, expected):
    assert run_conversion.parse_db_objects(raw) == expected


def _row(**overrides):
    fields = {
        "task": "fold_clothes",
        "operator": "op_7",
        "lab": "rl2",
        "scene": "kitchen_a",
        "objects": "{shirt,basket}",
    }
    return SimpleNamespace(**{**fields, **overrides})


def test_row_metadata_is_accepted():
    row = _row()
    meta = run_conversion.infer_required_metadata_from_row(row)
    assert meta["objects"] == ["shirt", "basket"]
    assert meta["missing_metadata"] == []
    assert run_conversion.required_metadata_row_error("fold_clothes", meta) is None


def test_null_objects_become_a_waiver():
    meta = run_conversion.infer_required_metadata_from_row(_row(objects="{None}"))
    assert meta["missing_metadata"] == ["objects"]
    assert run_conversion.required_metadata_row_error("fold_clothes", meta) is None


@pytest.mark.parametrize(
    "overrides, task_name, match",
    [
        ({"operator": "unknown"}, "fold_clothes", "operator"),
        ({"lab": "RL2"}, "fold_clothes", "lab"),
        ({}, "debug", "task_name"),
    ],
)
def test_bad_row_metadata_is_reported(overrides, task_name, match):
    meta = run_conversion.infer_required_metadata_from_row(_row(**overrides))
    assert match in run_conversion.required_metadata_row_error(task_name, meta)
