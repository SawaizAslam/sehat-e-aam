"""Tests for the trust scoring rule engine."""

from __future__ import annotations

from sehat.pipeline.trust_score import apply_trust_rules, compute_confidence
from sehat.schemas import FacilityType


def _empty_extraction() -> dict:
    return {
        "icu": {"present": "uncertain"},
        "ventilator": {"present": "uncertain"},
        "staff": {"anesthesiologist": "unknown", "surgeon": "unknown", "general_physician": "unknown", "specialist_types": []},
        "emergency": {"emergency_care": "uncertain", "is_24_7": False},
        "surgery": {"general_surgery": "uncertain", "caesarean": "uncertain", "cardiac": "uncertain"},
        "dialysis": {"present": "uncertain"},
    }


def test_clean_uncertain_extraction_does_not_floor_score() -> None:
    """Original guide flagged this as a near-zero; our fix should keep it moderate."""

    score, flags = apply_trust_rules(
        _empty_extraction(),
        facility_type=FacilityType.CLINIC,
        number_doctors=None,
        composite_text_length=300,
    )
    flag_ids = {f.flag_type for f in flags}
    assert "MISSING_CRITICAL_FIELDS" in flag_ids  # legitimately fires
    assert score > 0.5, f"score should not be punitively low for an uncertain clinic, got {score}"


def test_surgery_without_anesthesiologist_fires() -> None:
    ext = _empty_extraction()
    ext["surgery"]["general_surgery"] = "claimed"
    score, flags = apply_trust_rules(
        ext,
        facility_type=FacilityType.HOSPITAL,
        number_doctors=10,
        composite_text_length=600,
    )
    assert any(f.flag_type == "SURGERY_NO_ANESTHESIOLOGIST" for f in flags)
    assert score < 0.85


def test_caesarean_without_anesthesiologist_fires() -> None:
    ext = _empty_extraction()
    ext["surgery"]["caesarean"] = "confirmed"
    _, flags = apply_trust_rules(
        ext,
        facility_type=FacilityType.HOSPITAL,
        number_doctors=10,
        composite_text_length=600,
    )
    assert any(f.flag_type == "CAESAREAN_NO_ANESTHESIOLOGIST" for f in flags)


def test_cardiac_at_clinic_fires() -> None:
    ext = _empty_extraction()
    ext["surgery"]["cardiac"] = "claimed"
    _, flags = apply_trust_rules(
        ext,
        facility_type=FacilityType.CLINIC,
        number_doctors=2,
        composite_text_length=400,
    )
    assert any(f.flag_type == "CARDIAC_SURGERY_LOW_TIER" for f in flags)


def test_cardiac_at_hospital_does_not_fire() -> None:
    ext = _empty_extraction()
    ext["surgery"]["cardiac"] = "claimed"
    ext["staff"]["anesthesiologist"] = "full_time"
    ext["surgery"]["general_surgery"] = "uncertain"
    _, flags = apply_trust_rules(
        ext,
        facility_type=FacilityType.HOSPITAL,
        number_doctors=20,
        composite_text_length=1500,
    )
    assert not any(f.flag_type == "CARDIAC_SURGERY_LOW_TIER" for f in flags)


def test_ventilator_only_fires_when_icu_explicitly_absent() -> None:
    """Original guide fired this rule when ICU was uncertain - our fix requires explicit not_present."""

    ext_uncertain = _empty_extraction()
    ext_uncertain["ventilator"]["present"] = "claimed"
    _, flags_unc = apply_trust_rules(
        ext_uncertain,
        facility_type=FacilityType.HOSPITAL,
        number_doctors=10,
        composite_text_length=500,
    )
    assert not any(f.flag_type == "VENTILATOR_NO_ICU" for f in flags_unc)

    ext_absent = dict(ext_uncertain)
    ext_absent["icu"] = {"present": "not_present"}
    ext_absent["ventilator"]["present"] = "claimed"
    _, flags_abs = apply_trust_rules(
        ext_absent,
        facility_type=FacilityType.HOSPITAL,
        number_doctors=10,
        composite_text_length=500,
    )
    assert any(f.flag_type == "VENTILATOR_NO_ICU" for f in flags_abs)


def test_24_7_without_staff_requires_emergency_present() -> None:
    """is_24_7 alone shouldn't fire the rule; emergency must be present."""

    ext = _empty_extraction()
    ext["emergency"]["is_24_7"] = True
    _, flags = apply_trust_rules(
        ext,
        facility_type=FacilityType.HOSPITAL,
        number_doctors=5,
        composite_text_length=500,
    )
    assert not any(f.flag_type == "24_7_NO_STAFF" for f in flags)

    ext["emergency"]["emergency_care"] = "claimed"
    _, flags = apply_trust_rules(
        ext,
        facility_type=FacilityType.HOSPITAL,
        number_doctors=5,
        composite_text_length=500,
    )
    assert any(f.flag_type == "24_7_NO_STAFF" for f in flags)


def test_confidence_includes_completeness_and_consistency() -> None:
    ext = _empty_extraction()
    ext["icu"]["present"] = "confirmed"
    ext["staff"]["anesthesiologist"] = "full_time"
    ext["emergency"]["emergency_care"] = "confirmed"
    score, flags = apply_trust_rules(
        ext,
        facility_type=FacilityType.HOSPITAL,
        number_doctors=20,
        composite_text_length=2000,
    )
    confidence = compute_confidence(ext, flags=flags, composite_text_length=2000)
    assert 0.0 <= confidence.overall <= 1.0
    assert confidence.completeness >= 0.5  # 3 of 6 fields determined
    assert confidence.confidence_interval_low <= confidence.overall <= confidence.confidence_interval_high
