"""Pydantic v2 schemas for facility extraction and downstream artefacts.

Fixes applied vs the original guide:

* Pydantic v2 idioms throughout (``field_validator`` instead of ``validator``).
* ``AvailabilityStatus.NOT_PRESENT`` now means *explicitly* absent. Anything
  unmentioned is ``UNCERTAIN`` (the original conflated the two, which corrupted
  many downstream rules).
* ``FacilityType`` enum reflects the 6 actual values in the dataset, with
  ``farmacy`` typo normalised to ``PHARMACY``.
* ``FacilityExtraction.parse_relaxed`` accepts the LLM's raw JSON, fills
  defaults for missing keys, and coerces stringified ints (``"5"``) to int.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AvailabilityStatus(str, Enum):
    """Whether a capability exists at a facility.

    * ``confirmed``     - explicitly stated as functional / operational.
    * ``claimed``       - listed but no functional confirmation.
    * ``not_present``   - explicitly stated as absent.
    * ``uncertain``     - not mentioned, ambiguous, or unclear.
    """

    CONFIRMED = "confirmed"
    CLAIMED = "claimed"
    NOT_PRESENT = "not_present"
    UNCERTAIN = "uncertain"


class StaffType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    VISITING = "visiting"
    ON_CALL = "on_call"
    UNKNOWN = "unknown"


class FunctionalStatus(str, Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    PARTIALLY_FUNCTIONAL = "partially_functional"
    UNKNOWN = "unknown"


class FacilityType(str, Enum):
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    DENTIST = "dentist"
    DOCTOR = "doctor"
    PHARMACY = "pharmacy"
    UNKNOWN = "unknown"

    @classmethod
    def normalise(cls, raw: str | None) -> "FacilityType":
        """Normalise the raw ``facilityTypeId`` string (handles the ``farmacy`` typo)."""

        if not raw:
            return cls.UNKNOWN
        cleaned = raw.strip().lower()
        if cleaned in {"farmacy", "pharma", "chemist"}:
            return cls.PHARMACY
        try:
            return cls(cleaned)
        except ValueError:
            return cls.UNKNOWN


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Sub-profiles
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Base model that ignores extra keys and coerces sane defaults."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value else None  # NaN check
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in {"null", "none", "n/a", "nan"}:
            return None
        try:
            return int(float(cleaned))
        except ValueError:
            return None
    return None


class ICUProfile(_StrictModel):
    present: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    functional_status: FunctionalStatus = FunctionalStatus.UNKNOWN
    bed_count: int | None = None
    neonatal_icu: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    source_text: str | None = Field(default=None, max_length=2000)

    @field_validator("bed_count", mode="before")
    @classmethod
    def _bed_count_coerce(cls, v: Any) -> int | None:
        return _coerce_int(v)


class VentilatorProfile(_StrictModel):
    present: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    count: int | None = None
    reliability_note: str | None = None
    source_text: str | None = Field(default=None, max_length=2000)

    @field_validator("count", mode="before")
    @classmethod
    def _count_coerce(cls, v: Any) -> int | None:
        return _coerce_int(v)


class StaffProfile(_StrictModel):
    anesthesiologist: StaffType = StaffType.UNKNOWN
    surgeon: StaffType = StaffType.UNKNOWN
    general_physician: StaffType = StaffType.UNKNOWN
    specialist_types: list[str] = Field(default_factory=list)
    total_doctor_count: int | None = None
    source_text: str | None = Field(default=None, max_length=2000)

    @field_validator("total_doctor_count", mode="before")
    @classmethod
    def _doctor_coerce(cls, v: Any) -> int | None:
        return _coerce_int(v)

    @field_validator("specialist_types", mode="before")
    @classmethod
    def _specialist_coerce(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                return []
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                return [s.strip() for s in cleaned.split(",") if s.strip()]
        return []


class EmergencyProfile(_StrictModel):
    emergency_care: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    is_24_7: bool = False
    ambulance: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    trauma_capability: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    source_text: str | None = Field(default=None, max_length=2000)


class SurgeryProfile(_StrictModel):
    general_surgery: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    appendectomy: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    caesarean: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    orthopedic: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    cardiac: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    source_text: str | None = Field(default=None, max_length=2000)


class DialysisProfile(_StrictModel):
    present: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    machine_count: int | None = None
    source_text: str | None = Field(default=None, max_length=2000)

    @field_validator("machine_count", mode="before")
    @classmethod
    def _machine_coerce(cls, v: Any) -> int | None:
        return _coerce_int(v)


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------


class FacilityExtraction(_StrictModel):
    facility_id: str
    icu: ICUProfile = Field(default_factory=ICUProfile)
    ventilator: VentilatorProfile = Field(default_factory=VentilatorProfile)
    staff: StaffProfile = Field(default_factory=StaffProfile)
    emergency: EmergencyProfile = Field(default_factory=EmergencyProfile)
    surgery: SurgeryProfile = Field(default_factory=SurgeryProfile)
    dialysis: DialysisProfile = Field(default_factory=DialysisProfile)
    specialties_extracted: list[str] = Field(default_factory=list)
    extraction_notes: str | None = None
    raw_text_used: str = ""

    @field_validator("specialties_extracted", mode="before")
    @classmethod
    def _spec_coerce(cls, v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return []

    @classmethod
    def parse_relaxed(cls, payload: dict[str, Any], facility_id: str) -> "FacilityExtraction":
        """Validate an LLM output dict, filling defaults for missing sections."""

        merged = {"facility_id": facility_id}
        merged.update(payload or {})
        merged["facility_id"] = facility_id  # never trust the LLM with the id
        return cls.model_validate(merged)


# ---------------------------------------------------------------------------
# Trust + confidence + gold record
# ---------------------------------------------------------------------------


class TrustFlag(_StrictModel):
    flag_type: str
    severity: Severity
    description: str
    supporting_evidence: str = ""


class ConfidenceScore(_StrictModel):
    completeness: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    reliability: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    confidence_interval_low: float = Field(ge=0.0, le=1.0)
    confidence_interval_high: float = Field(ge=0.0, le=1.0)


class FacilityGoldRecord(_StrictModel):
    facility_id: str
    name: str
    address_city: str | None = None
    address_state: str | None = None
    address_zip: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    facility_type: FacilityType = FacilityType.UNKNOWN
    operator_type: str | None = None
    extraction: FacilityExtraction
    trust_score: float = Field(ge=0.0, le=1.0)
    trust_flags: list[TrustFlag] = Field(default_factory=list)
    confidence: ConfidenceScore
    correction_iterations: int = 0
    embedding_text: str
    extraction_version: str = "1.0"


class MedicalDesertReport(_StrictModel):
    pin_code: str
    state: str
    facility_count: int
    icu_coverage: float = Field(ge=0.0, le=1.0)
    dialysis_coverage: float = Field(ge=0.0, le=1.0)
    emergency_coverage: float = Field(ge=0.0, le=1.0)
    surgery_coverage: float = Field(ge=0.0, le=1.0)
    desert_risk_score: float = Field(ge=0.0, le=1.0)
    desert_categories: list[str] = Field(default_factory=list)
    avg_trust_score: float
    centroid_lat: float | None = None
    centroid_lon: float | None = None
    is_high_risk: bool = False


# ---------------------------------------------------------------------------
# Reasoning engine response
# ---------------------------------------------------------------------------


class RankedResult(_StrictModel):
    rank: int
    facility_id: str
    facility_name: str
    suitability_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    matched_capabilities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class ReasoningResponse(_StrictModel):
    query_interpretation: str = ""
    ranked_results: list[RankedResult] = Field(default_factory=list)
    recommendation_summary: str = ""
    uncertainty_note: str = ""


__all__ = [
    "AvailabilityStatus",
    "StaffType",
    "FunctionalStatus",
    "FacilityType",
    "Severity",
    "ICUProfile",
    "VentilatorProfile",
    "StaffProfile",
    "EmergencyProfile",
    "SurgeryProfile",
    "DialysisProfile",
    "FacilityExtraction",
    "TrustFlag",
    "ConfidenceScore",
    "FacilityGoldRecord",
    "MedicalDesertReport",
    "RankedResult",
    "ReasoningResponse",
]
