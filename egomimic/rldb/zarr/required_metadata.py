"""
Metadata required in every contributed zarr episode.

See CONTRIBUTING_DATA.md §6.3. ZarrWriter takes each REQUIRED_METADATA_FIELDS
entry as a keyword argument with no default and checks it in write(); the
episode validator (egomimic.rldb.zarr.validate) applies the same rules to a
finished store. The only way to leave a field out is an explicit
MissingMetadata().
"""

import argparse
import re
from collections.abc import Iterable
from typing import Any

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

_LAB_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_VALUES


class MissingMetadata:
    """Explicit opt-out for a required metadata field the source data does not have.

    Pass it in place of the value, e.g. ``scene=MissingMetadata()``. ``None``,
    ``""`` and placeholders like ``"unknown"`` are rejected, so a field can only
    be left out on purpose. The field is stored as ``null`` in zarr.attrs.

    Only use this if absolutely necessary, i.e. the source data truly has no
    way to recover the value. Episodes with missing metadata are much harder
    to filter and train on, and fail validation without --allow-missing.
    """

    def __repr__(self) -> str:
        return "MissingMetadata()"


def objects_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """``objects`` as a list: a str is a comma-separated list of names."""
    if isinstance(value, str):
        return [o.strip() for o in value.split(",")]
    return list(value)


def required_metadata_error(field: str, value: Any) -> str | None:
    """Return why ``value`` is not acceptable for required metadata ``field``, or None.

    Shared by ZarrWriter.write and the episode validator so both
    apply the same rules. ``value`` is a concrete value, not MissingMetadata.
    """
    hint = "See CONTRIBUTING_DATA.md §6.3."
    if field == "objects" and isinstance(value, (list, tuple, str)):
        if isinstance(value, str) and value != value.strip():
            return f"objects={value!r} has leading/trailing whitespace. {hint}"
        names = objects_list(value)
        if not names or names == [""]:
            return (
                f"objects must name at least one object. If the source data has "
                f"no objects, pass objects=MissingMetadata(). {hint}"
            )
        for obj in names:
            if not isinstance(obj, str) or is_placeholder(obj):
                return f"objects entries must be real names, got {obj!r}. {hint}"
            if obj != obj.strip():
                return f"objects entry {obj!r} has leading/trailing whitespace. {hint}"
        return None
    if not isinstance(value, str):
        extra = " or list[str]" if field == "objects" else ""
        return (
            f"{field} must be a str{extra}, got {type(value).__name__} ({value!r}). "
            f"If the source data has no {field}, pass "
            f"{field}=MissingMetadata(). {hint}"
        )
    if is_placeholder(value):
        return (
            f"{field}={value!r} is a placeholder, not a value. If the source data "
            f"has no {field}, pass {field}=MissingMetadata(). {hint}"
        )
    if value != value.strip():
        return f"{field}={value!r} has leading/trailing whitespace. {hint}"
    if field == "lab" and not _LAB_RE.fullmatch(value):
        return (
            f"lab={value!r} must be a short, stable, lowercase identifier "
            f"(letters, digits, '_' or '-'), e.g. 'mecka'. {hint}"
        )
    return None


def check_stored_metadata(
    meta: dict[str, Any], *, allow_missing: bool = False
) -> tuple[list[str], list[str], list[str]]:
    """(errors, warnings, passed field names) for a finished episode's metadata.

    Shared by the zarr and MCAP validators. An absent key is always an error; a
    ``None`` value is an explicit waiver (MissingMetadata), an error unless
    ``allow_missing``. Concrete values go through ``required_metadata_error``.
    """
    errors, warnings, passed = [], [], []
    for name in REQUIRED_METADATA_FIELDS:
        if name not in meta:
            errors.append(
                f"Missing metadata field: {name}. See CONTRIBUTING_DATA.md §6.3."
            )
        elif meta[name] is None:
            msg = f"{name}: marked as not available (MissingMetadata)"
            if allow_missing:
                warnings.append(msg)
            else:
                errors.append(
                    msg + ". We heavily recommend providing it; pass "
                    "--allow-missing to accept the episode anyway."
                )
        elif (err := required_metadata_error(name, meta[name])) is not None:
            errors.append(err)
        else:
            passed.append(name)
    return errors, warnings, passed


def add_required_metadata_args(parser: argparse.ArgumentParser) -> None:
    """Add --operator/--lab/--scene/--objects/--missing-metadata to a converter CLI.

    The converter supplies its own --task-name. None of these have usable
    defaults: an omitted field fails in ZarrWriter.write unless it is waived
    with ``--missing-metadata FIELD``.
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
        choices=REQUIRED_METADATA_FIELDS,
        metavar="FIELD",
        help="Explicitly mark FIELD as not available in the source data. "
        "Repeatable. Only use this if absolutely necessary: provide the value "
        "whenever the source data has it.",
    )


def _build_required_metadata(
    values: dict[str, Any], missing: Iterable[str] = ()
) -> dict[str, Any]:
    """Merge plain values and the names of waived fields into ZarrWriter kwargs.

    Both inputs are plain data so they can travel through Ray / argparse.
    A field named in ``missing`` becomes MissingMetadata(); giving a field both
    a value and a waiver is an error.
    """
    if errors := waiver_errors(values, missing):
        raise ValueError(errors[0])
    missing = set(missing)
    return {
        field: MissingMetadata() if field in missing else values.get(field)
        for field in REQUIRED_METADATA_FIELDS
    }


def waiver_errors(values: dict[str, Any], missing: Iterable[str]) -> list[str]:
    """Unknown waived field names, and fields with both a value and a waiver."""
    missing = set(missing)
    errors = []
    unknown = missing - set(REQUIRED_METADATA_FIELDS)
    if unknown:
        errors.append(
            f"missing_metadata names unknown fields {sorted(unknown)}; "
            f"expected a subset of {list(REQUIRED_METADATA_FIELDS)}"
        )
    for field in REQUIRED_METADATA_FIELDS:
        value = values.get(field)
        empty = value is None or (hasattr(value, "__len__") and len(value) == 0)
        if field in missing and not empty:
            errors.append(
                f"{field} has both a value ({value!r}) and is marked missing; "
                "pass one or the other"
            )
    return errors


def check_writer_metadata(kwargs: dict[str, Any]) -> None:
    """Raise ValueError on the first unacceptable value in ZarrWriter metadata kwargs.

    The same check ZarrWriter.write runs; call it before an expensive extraction
    so a bad value fails fast.
    """
    for field in REQUIRED_METADATA_FIELDS:
        value = kwargs.get(field)
        if isinstance(value, MissingMetadata):
            continue
        if (err := required_metadata_error(field, value)) is not None:
            raise ValueError(err)


def required_metadata_from_args(args: Any) -> dict[str, Any]:
    """ZarrWriter required-metadata kwargs from an argparse/SimpleNamespace.

    ``args.missing_metadata`` is an iterable of waived field names (repeated
    --missing-metadata on the CLI, or a list from run_conversion).
    """
    missing = getattr(args, "missing_metadata", None) or ()
    values = {f: getattr(args, f, None) for f in REQUIRED_METADATA_FIELDS}
    return _build_required_metadata(values, missing)
