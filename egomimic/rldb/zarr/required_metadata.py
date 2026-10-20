"""
Metadata required in every contributed zarr episode.

See CONTRIBUTING_DATA.md §6.3. ZarrWriter.create_and_write requires each
REQUIRED_METADATA_FIELDS entry as a keyword argument with no default, and the episode
validator (egomimic.rldb.zarr.validate) applies the same rules to a finished
store. The only way to leave a field out is an explicit MissingMetadata(reason).
"""

import argparse
import re
from typing import Any

# Metadata every contributed episode must carry in zarr.attrs (see
# CONTRIBUTING_DATA.md §6.3). create_and_write requires each one as a keyword
# argument with no default; the only way to leave one out is an explicit
# MissingMetadata(reason).
REQUIRED_METADATA_FIELDS = ("task_name", "operator", "lab", "scene", "objects")

# Values that mean "I didn't fill this in". Rejected so a placeholder can't
# stand in for a real value (compared case-insensitively after strip).
_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "unknown",
        "none",
        "null",
        "n/a",
        "na",
        "tbd",
        "todo",
        "debug",
        "test",
        "placeholder",
        "-",
        "?",
    }
)

_LAB_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_VALUES


class MissingMetadata:
    """Explicit opt-out for one required metadata field the source data does not have.

    Pass it in place of the value, e.g.
    ``operator=MissingMetadata("Vendor export has no per-episode collector ID")``.
    ``None``, ``""`` and placeholders like ``"unknown"`` are rejected, so a field
    can only be left out on purpose. The field is stored as ``None`` and the
    reason under ``zarr.attrs["missing_metadata"][field]``.
    """

    MIN_REASON_LEN = 15

    def __init__(self, reason: str):
        if (
            not isinstance(reason, str)
            or len(reason.strip()) < self.MIN_REASON_LEN
            or _is_placeholder(reason)
        ):
            raise ValueError(
                "MissingMetadata needs a real reason (at least "
                f"{self.MIN_REASON_LEN} characters) explaining why the source data "
                f"has no value for this field, got {reason!r}. "
                "See CONTRIBUTING_DATA.md §6.3."
            )
        self.reason = reason.strip()

    def __repr__(self) -> str:
        return f"MissingMetadata({self.reason!r})"


def required_metadata_error(field: str, value: Any) -> str | None:
    """Return why ``value`` is not acceptable for required metadata ``field``, or None.

    Shared by ZarrWriter.create_and_write and the episode validator so both
    apply the same rules. ``value`` is a concrete value, not MissingMetadata.
    """
    hint = "See CONTRIBUTING_DATA.md §6.3."
    if field == "objects" and isinstance(value, (list, tuple)):
        if not value:
            return f"objects must not be an empty list. {hint}"
        for obj in value:
            if not isinstance(obj, str) or _is_placeholder(obj):
                return f"objects entries must be real names, got {obj!r}. {hint}"
        return None
    if not isinstance(value, str):
        extra = " or list[str]" if field == "objects" else ""
        return (
            f"{field} must be a str{extra}, got {type(value).__name__} ({value!r}). "
            f"If the source data has no {field}, pass "
            f'{field}=MissingMetadata("<why>"). {hint}'
        )
    if _is_placeholder(value):
        return (
            f"{field}={value!r} is a placeholder, not a value. If the source data "
            f'has no {field}, pass {field}=MissingMetadata("<why>"). {hint}'
        )
    if field == "lab" and not _LAB_RE.match(value):
        return (
            f"lab={value!r} must be a short, stable, lowercase identifier "
            f"(letters, digits, '_' or '-'), e.g. 'mecka'. {hint}"
        )
    return None


def add_required_metadata_args(parser: argparse.ArgumentParser) -> None:
    """Add --operator/--lab/--scene/--objects/--missing-metadata to a converter CLI.

    The converter supplies its own --task-name. None of these have usable
    defaults: an omitted field fails in create_and_write unless it is waived
    with ``--missing-metadata FIELD="reason"``.
    """
    group = parser.add_argument_group(
        "required episode metadata (CONTRIBUTING_DATA.md §6.3)"
    )
    group.add_argument("--operator", type=str, help="ID of the collecting operator")
    group.add_argument("--lab", type=str, help="Short lowercase source id, e.g. rl2")
    group.add_argument("--scene", type=str, help="Scene / location label")
    group.add_argument("--objects", type=str, help="Objects manipulated")
    group.add_argument(
        "--missing-metadata",
        action="append",
        default=[],
        metavar='FIELD="REASON"',
        help="Explicitly record that the source data has no value for FIELD. "
        "Repeatable. Strongly discouraged: provide the value if at all possible.",
    )


def build_required_metadata(
    values: dict[str, Any], missing: dict[str, str] | None = None
) -> dict[str, Any]:
    """Merge plain values and {field: reason} waivers into create_and_write kwargs.

    Both inputs are plain data so they can travel through Ray / argparse.
    A field in ``missing`` becomes MissingMetadata(reason); giving a field both
    a value and a waiver is an error.
    """
    missing = dict(missing or {})
    unknown = set(missing) - set(REQUIRED_METADATA_FIELDS)
    if unknown:
        raise ValueError(
            f"missing_metadata names unknown fields {sorted(unknown)}; "
            f"expected a subset of {list(REQUIRED_METADATA_FIELDS)}"
        )
    out = {}
    for field in REQUIRED_METADATA_FIELDS:
        value = values.get(field)
        if field in missing:
            if value not in (None, ""):
                raise ValueError(
                    f"{field} has both a value ({value!r}) and a missing_metadata "
                    "reason; pass one or the other"
                )
            out[field] = MissingMetadata(missing[field])
        else:
            out[field] = value
    return out


def parse_missing_metadata(items: list[str] | None) -> dict[str, str]:
    """Parse repeated ``FIELD=REASON`` CLI strings into {field: reason}."""
    missing = {}
    for item in items or []:
        field, sep, reason = item.partition("=")
        if not sep:
            raise ValueError(f"--missing-metadata expects FIELD=REASON, got {item!r}")
        missing[field.strip()] = reason.strip()
    return missing


def required_metadata_from_args(args: Any) -> dict[str, Any]:
    """create_and_write required-metadata kwargs from an argparse/SimpleNamespace.

    ``args.missing_metadata`` may be a list of FIELD=REASON strings (CLI) or a
    {field: reason} dict (programmatic callers such as run_conversion).
    """
    missing = getattr(args, "missing_metadata", None)
    if not isinstance(missing, dict):
        missing = parse_missing_metadata(missing)
    values = {f: getattr(args, f, None) for f in REQUIRED_METADATA_FIELDS}
    return build_required_metadata(values, missing)
