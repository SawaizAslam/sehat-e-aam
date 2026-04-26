"""Smoke tests for Pydantic schemas + their relaxed parser."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sehat.schemas import (
    AvailabilityStatus,
    FacilityExtraction,
    FacilityType,
    FunctionalStatus,
    StaffType,
)


def test_facility_type_normalises_typo() -> None:
    assert FacilityType.normalise("farmacy") == FacilityType.PHARMACY
    assert FacilityType.normalise("pharmacy") == FacilityType.PHARMACY
    assert FacilityType.normalise(None) == FacilityType.UNKNOWN
    assert FacilityType.normalise("CLINIC") == FacilityType.CLINIC
    assert FacilityType.normalise("not-a-thing") == FacilityType.UNKNOWN


def test_parse_relaxed_fills_defaults_and_keeps_id() -> None:
    payload = {
        "icu": {"present": "claimed", "bed_count": "12"},
        "staff": {"specialist_types": "[\"cardiology\", \"oncology\"]"},
    }
    e = FacilityExtraction.parse_relaxed(payload, facility_id="abc123")
    assert e.facility_id == "abc123"
    assert e.icu.present == AvailabilityStatus.CLAIMED
    assert e.icu.bed_count == 12  # coerced from string
    assert e.staff.specialist_types == ["cardiology", "oncology"]
    assert e.surgery.general_surgery == AvailabilityStatus.UNCERTAIN
    assert e.staff.surgeon == StaffType.UNKNOWN
    assert e.icu.functional_status == FunctionalStatus.UNKNOWN


def test_parse_relaxed_rejects_invalid_enum() -> None:
    with pytest.raises(ValidationError):
        FacilityExtraction.parse_relaxed(
            {"icu": {"present": "definitely_yes"}},
            facility_id="x",
        )


def test_facility_id_overridden_even_if_llm_lies() -> None:
    payload = {"facility_id": "wrong-id-from-llm", "icu": {"present": "confirmed"}}
    e = FacilityExtraction.parse_relaxed(payload, facility_id="trusted-id")
    assert e.facility_id == "trusted-id"


def test_specialist_types_coerce_csv() -> None:
    payload = {"staff": {"specialist_types": "cardiology, oncology, , urology"}}
    e = FacilityExtraction.parse_relaxed(payload, facility_id="abc")
    assert e.staff.specialist_types == ["cardiology", "oncology", "urology"]


def test_int_coercion_handles_nan_strings() -> None:
    payload = {"icu": {"present": "claimed", "bed_count": "nan"}}
    e = FacilityExtraction.parse_relaxed(payload, facility_id="abc")
    assert e.icu.bed_count is None
