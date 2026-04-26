# Agentic Healthcare Intelligence System
## Complete Production-Grade Backend Implementation Guide
### Databricks Free Edition · Mosaic AI Vector Search · MLflow 3 · Delta Tables

---

> **How to use this guide in Cursor:** Open this file as your project root reference. Each section maps to a separate Databricks notebook. Code cells are marked with `# CELL N` headers — paste them sequentially into notebook cells. Run notebooks in the order: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Dataset Schema & Pydantic Models](#2-dataset-schema--pydantic-models)
3. [Notebook 01: Data Ingestion Layer](#3-notebook-01-data-ingestion-layer)
4. [Notebook 02: LLM Extraction Layer](#4-notebook-02-llm-extraction-layer)
5. [Notebook 03: Trust Scoring & Validation](#5-notebook-03-trust-scoring--validation)
6. [Notebook 04: Self-Correction Agent Loop](#6-notebook-04-self-correction-agent-loop)
7. [Notebook 05: Vector Search Setup](#7-notebook-05-vector-search-setup)
8. [Notebook 06: Multi-Attribute Reasoning Engine](#8-notebook-06-multi-attribute-reasoning-engine)
9. [Notebook 07: Medical Desert Detection](#9-notebook-07-medical-desert-detection)
10. [Notebook 08: API Serving Layer](#10-notebook-08-api-serving-layer)
11. [MLflow 3 Tracing Integration](#11-mlflow-3-tracing-integration)
12. [LLM Prompts Reference](#12-llm-prompts-reference)
13. [Confidence Scoring Model](#13-confidence-scoring-model)
14. [Crisis Mapping Data Pipeline](#14-crisis-mapping-data-pipeline)

---

## 1. Architecture Overview

### 1.1 Full System ASCII Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                  AGENTIC HEALTHCARE INTELLIGENCE SYSTEM                          ║
║                      Databricks Free Edition                                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA INGESTION                                                        │
│                                                                                  │
│  CSV/XLSX Upload ──► Databricks DBFS ──► Bronze Delta Table                    │
│  (10k rows: name, address, description, specialties,                             │
│   equipment, capability, lat/lon, zip, numberDoctors)                           │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: LLM EXTRACTION (Notebook 02)                                          │
│                                                                                  │
│  Bronze Delta ──► [EXTRACTOR AGENT]                                             │
│                        │                                                         │
│                        ├── System Prompt + Pydantic Schema                      │
│                        ├── Batch processing (50 rows/call)                      │
│                        └── Output: FacilityExtraction JSON                      │
│                              │                                                   │
│                              ▼                                                   │
│                    Silver Delta Table                                            │
│                    (structured extracted fields)                                 │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │
                        ┌───────┴────────┐
                        ▼                ▼
┌───────────────────────────┐  ┌─────────────────────────────────────────────────┐
│  LAYER 3A: TRUST SCORING  │  │  LAYER 3B: SELF-CORRECTION LOOP                 │
│                           │  │                                                   │
│  Rule-Based Engine:       │  │  Extractor ──► Validator ──► Corrector           │
│  - Surgery w/o anesthesia │  │      ▲               │                           │
│  - ICU non-functional     │  │      └───────────────┘                           │
│  - 24/7 w/o staff         │  │      (max 2 iterations)                          │
│  - Bed count mismatches   │  │                                                   │
│                           │  │  MLflow Trace logged per facility                │
│  LLM Contradiction Check  │  └─────────────────────────────────────────────────┘
│  Trust Score: 0.0–1.0     │
└───────────────┬───────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: GOLD DELTA TABLE (unified enriched store)                             │
│                                                                                  │
│  facility_id │ extracted_json │ trust_score │ confidence_json │ flags │ embeddings_text │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
┌──────────────────────────┐   ┌────────────────────────────────────────────────┐
│  LAYER 5: VECTOR SEARCH  │   │  LAYER 6: AGGREGATION (Medical Deserts)        │
│                          │   │                                                  │
│  Mosaic AI Vector Search │   │  GROUP BY pin_code / state / district           │
│  Index on Gold Table     │   │  Score: ICU coverage, dialysis, emergency      │
│  text-embedding-3-small  │   │  Output: desert_risk_score per region          │
│  (or bge-large via DBRX) │   │  Geo-ready JSON for map overlay               │
└──────────────┬───────────┘   └────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 7: REASONING ENGINE                                                      │
│                                                                                  │
│  Query ──► [Vector Retrieval (top-20)]                                          │
│               ──► [Structured Filter (Delta SQL)]                               │
│                     ──► [LLM Ranker + Explainer]                               │
│                           ──► Ranked Results + Chain of Thought                 │
│                                 ──► MLflow Trace                               │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 8: SERVING API                                                           │
│                                                                                  │
│  query_facilities(query, filters) ──► ranked results + citations               │
│  get_trust_report(facility_id)    ──► trust breakdown + flags                  │
│  get_desert_map(state, criteria)  ──► geo-tagged risk regions                  │
│  get_facility_profile(id)         ──► full enriched profile                    │
└─────────────────────────────────────────────────────────────────────────────────┘

OBSERVABILITY PLANE (cross-cutting):
  MLflow 3 Traces ──► Every agent call logged with: prompt, output, latency, tokens
  Delta Table audit_log ──► Every extraction + correction event recorded
```

### 1.2 Delta Table Lineage

```
Bronze (raw ingest)
  └─► Silver (LLM extracted, validated)
        └─► Gold (trust-scored, embedding-ready, corrected)
              ├─► Vector Search Index
              └─► Desert Aggregation Table
```

### 1.3 Databricks Free Edition Constraints & Mitigations

| Constraint | Mitigation |
|---|---|
| No GPU cluster | Use Databricks Serverless SQL + Foundation Model API endpoints |
| Limited DBUs | Batch LLM calls (50 rows/batch), async where possible |
| No dedicated Vector Search cluster | Use Databricks Managed Vector Search (serverless, included in free tier) |
| Rate limits on Foundation Model API | Exponential backoff, batch size 10–50 rows |
| Storage limits | Use Delta with `OPTIMIZE` + `ZORDER` to reduce footprint |

---

## 2. Dataset Schema & Pydantic Models

### 2.1 Raw Dataset Columns (from VF_Hackathon_Dataset_India_Large.xlsx)

```
name, phone_numbers, officialPhone, email, websites, officialWebsite,
yearEstablished, address_line1, address_line2, address_line3,
address_city, address_stateOrRegion, address_zipOrPostcode, address_country,
facilityTypeId, operatorTypeId, affiliationTypeIds,
description,                          # PRIMARY UNSTRUCTURED TEXT
numberDoctors, capacity,
specialties,                          # JSON array string
procedure,                            # JSON array string
equipment,                            # JSON array string
capability,                           # JSON array string  ← KEY TARGET
recency_of_page_update,
distinct_social_media_presence_count,
affiliated_staff_presence, custom_logo_presence,
number_of_facts_about_the_organization,
post_metrics_most_recent_social_media_post_date, post_metrics_post_count,
engagement_metrics_n_followers, engagement_metrics_n_likes,
latitude, longitude
```

### 2.2 Core Pydantic Schemas

```python
# schemas.py
# Place this file at: /Workspace/Users/<your-user>/healthcare_agent/schemas.py

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from enum import Enum

# ─── Enums ────────────────────────────────────────────────────────────────────

class AvailabilityStatus(str, Enum):
    CONFIRMED = "confirmed"
    CLAIMED = "claimed"
    UNCERTAIN = "uncertain"
    NOT_PRESENT = "not_present"

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

# ─── Sub-schemas ──────────────────────────────────────────────────────────────

class ICUProfile(BaseModel):
    present: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    functional_status: FunctionalStatus = FunctionalStatus.UNKNOWN
    bed_count: Optional[int] = None
    neonatal_icu: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    source_text: Optional[str] = Field(
        None,
        description="Exact substring from notes supporting this extraction"
    )

class VentilatorProfile(BaseModel):
    present: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    count: Optional[int] = None
    reliability_note: Optional[str] = None
    source_text: Optional[str] = None

class StaffProfile(BaseModel):
    anesthesiologist: StaffType = StaffType.UNKNOWN
    surgeon: StaffType = StaffType.UNKNOWN
    general_physician: StaffType = StaffType.UNKNOWN
    specialist_types: List[str] = Field(default_factory=list)
    total_doctor_count: Optional[int] = None
    source_text: Optional[str] = None

class EmergencyProfile(BaseModel):
    emergency_care: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    is_24_7: bool = False
    ambulance: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    trauma_capability: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    source_text: Optional[str] = None

class SurgeryProfile(BaseModel):
    general_surgery: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    appendectomy: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    caesarean: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    orthopedic: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    cardiac: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    source_text: Optional[str] = None

class DialysisProfile(BaseModel):
    present: AvailabilityStatus = AvailabilityStatus.UNCERTAIN
    machine_count: Optional[int] = None
    source_text: Optional[str] = None

class ConfidenceScore(BaseModel):
    completeness: float = Field(ge=0.0, le=1.0, description="Fraction of fields populated")
    consistency: float = Field(ge=0.0, le=1.0, description="Internal logical consistency")
    reliability: float = Field(ge=0.0, le=1.0, description="Source credibility signal")
    overall: float = Field(ge=0.0, le=1.0, description="Weighted composite score")
    confidence_interval_low: float = Field(ge=0.0, le=1.0)
    confidence_interval_high: float = Field(ge=0.0, le=1.0)

class TrustFlag(BaseModel):
    flag_type: str  # e.g. "SURGERY_WITHOUT_ANESTHESIOLOGIST"
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    description: str
    supporting_evidence: str

# ─── Primary Extraction Schema ────────────────────────────────────────────────

class FacilityExtraction(BaseModel):
    facility_id: str
    icu: ICUProfile = Field(default_factory=ICUProfile)
    ventilator: VentilatorProfile = Field(default_factory=VentilatorProfile)
    staff: StaffProfile = Field(default_factory=StaffProfile)
    emergency: EmergencyProfile = Field(default_factory=EmergencyProfile)
    surgery: SurgeryProfile = Field(default_factory=SurgeryProfile)
    dialysis: DialysisProfile = Field(default_factory=DialysisProfile)
    specialties_extracted: List[str] = Field(default_factory=list)
    extraction_notes: Optional[str] = Field(
        None,
        description="Free-text notes about ambiguous or uncertain extractions"
    )
    raw_text_used: str = Field(description="The exact input text this was extracted from")

class FacilityGoldRecord(BaseModel):
    """Final enriched record written to Gold Delta Table."""
    facility_id: str
    name: str
    address_city: Optional[str]
    address_stateOrRegion: Optional[str]
    address_zipOrPostcode: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    facilityTypeId: Optional[str]
    operatorTypeId: Optional[str]
    extraction: FacilityExtraction
    trust_score: float = Field(ge=0.0, le=1.0)
    trust_flags: List[TrustFlag] = Field(default_factory=list)
    confidence: ConfidenceScore
    correction_iterations: int = Field(default=0)
    embedding_text: str = Field(description="Concatenated text used for vector embedding")
    extraction_version: str = "1.0"

class MedicalDesertReport(BaseModel):
    pin_code: str
    state: str
    district: Optional[str]
    facility_count: int
    icu_coverage: float = Field(ge=0.0, le=1.0, description="Fraction of facilities with confirmed ICU")
    dialysis_coverage: float = Field(ge=0.0, le=1.0)
    emergency_coverage: float = Field(ge=0.0, le=1.0)
    surgery_coverage: float = Field(ge=0.0, le=1.0)
    desert_risk_score: float = Field(ge=0.0, le=1.0, description="1.0 = complete desert")
    desert_categories: List[str] = Field(description="Which capabilities are desert-level")
    avg_trust_score: float
    centroid_lat: Optional[float]
    centroid_lon: Optional[float]
    is_high_risk: bool
```

---

## 3. Notebook 01: Data Ingestion Layer

**Notebook name:** `01_data_ingestion`
**Cluster:** Serverless (Databricks Free Edition)

```python
# CELL 1 — Install dependencies
%pip install pydantic>=2.0 pandas openpyxl

# CELL 2 — Configuration block (edit these before running)
CATALOG = "main"               # Unity Catalog name (default on free tier)
SCHEMA = "healthcare"          # Schema/database name
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.facilities_bronze"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.facilities_silver"
GOLD_TABLE   = f"{CATALOG}.{SCHEMA}.facilities_gold"
AUDIT_TABLE  = f"{CATALOG}.{SCHEMA}.audit_log"
DESERT_TABLE = f"{CATALOG}.{SCHEMA}.medical_deserts"

# Source: upload your CSV to DBFS first
# In the Databricks UI: Data > Add Data > Upload File
# Then reference it here:
CSV_PATH = "/dbfs/FileStore/vf_hackathon_dataset_india_large.csv"

# Foundation Model API — uses Databricks-hosted models (free tier)
# Options: databricks-dbrx-instruct, databricks-meta-llama-3-1-70b-instruct
LLM_ENDPOINT = "databricks-meta-llama-3-1-70b-instruct"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"

# CELL 3 — Create catalog schema if not exists
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# CELL 4 — Ingest CSV to Bronze Delta Table
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import json

# Read CSV — force all columns to string first for safe ingest
df_raw = spark.read.option("header", "true") \
    .option("inferSchema", "false") \
    .option("multiLine", "true") \
    .option("escape", '"') \
    .csv(CSV_PATH)

# Add ingestion metadata
df_bronze = df_raw.withColumn("_ingest_ts", F.current_timestamp()) \
    .withColumn("facility_id", F.sha2(F.col("name").cast(StringType()), 256))

# Write to Bronze (append-safe, idempotent with MERGE later)
df_bronze.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(BRONZE_TABLE)

print(f"✅ Bronze table written: {df_bronze.count()} rows → {BRONZE_TABLE}")

# CELL 5 — Build composite text field for extraction
# This is the primary input to the LLM extraction agent
df_text = spark.table(BRONZE_TABLE).select(
    "facility_id", "name",
    "address_city", "address_stateOrRegion", "address_zipOrPostcode",
    "facilityTypeId", "operatorTypeId",
    "description", "specialties", "procedure", "equipment", "capability",
    "numberDoctors", "capacity", "latitude", "longitude",
    "_ingest_ts"
).withColumn(
    "composite_text",
    F.concat_ws(
        " | ",
        F.coalesce(F.col("name"), F.lit("")),
        F.coalesce(F.col("description"), F.lit("")),
        F.coalesce(F.col("specialties"), F.lit("")),
        F.coalesce(F.col("procedure"), F.lit("")),
        F.coalesce(F.col("equipment"), F.lit("")),
        F.coalesce(F.col("capability"), F.lit(""))
    )
).withColumn(
    "composite_text_length",
    F.length(F.col("composite_text"))
)

df_text.createOrReplaceTempView("facilities_for_extraction")

# CELL 6 — Validate ingestion quality
print("=== INGESTION QUALITY REPORT ===")
total = df_text.count()
print(f"Total records: {total}")
print(f"Records with description: {df_text.filter(F.col('description').isNotNull()).count()}")
print(f"Records with equipment: {df_text.filter(F.col('equipment') != '[]').count()}")
print(f"Records with lat/lon: {df_text.filter(F.col('latitude').isNotNull()).count()}")
print(f"Records with zip: {df_text.filter(F.col('address_zipOrPostcode').isNotNull()).count()}")

# Show state distribution
display(df_text.groupBy("address_stateOrRegion").count().orderBy(F.desc("count")).limit(20))

# CELL 7 — Create audit log table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {AUDIT_TABLE} (
        event_id STRING,
        facility_id STRING,
        event_type STRING,
        event_ts TIMESTAMP,
        agent_name STRING,
        iteration INT,
        input_hash STRING,
        output_hash STRING,
        llm_model STRING,
        tokens_used INT,
        latency_ms FLOAT,
        notes STRING
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print(f"✅ Audit table ready: {AUDIT_TABLE}")
```

---

## 4. Notebook 02: LLM Extraction Layer

**Notebook name:** `02_llm_extraction`

```python
# CELL 1 — Imports and setup
%pip install pydantic>=2.0 mlflow>=3.0 openai

import mlflow
import mlflow.tracking
from mlflow.entities import SpanType
import json
import time
import hashlib
from typing import List, Optional, Dict, Any
from pyspark.sql import functions as F, Row
from pyspark.sql.types import StringType, StructType, StructField, FloatType
import requests
import os

# Run config (shared from Notebook 01 via widget or re-declared)
CATALOG = "main"
SCHEMA = "healthcare"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.facilities_bronze"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.facilities_silver"
LLM_ENDPOINT = "databricks-meta-llama-3-1-70b-instruct"

# MLflow experiment
mlflow.set_experiment("/Users/<your-email>/healthcare_extraction")

# CELL 2 — Databricks Foundation Model API wrapper
import os

def call_llm(
    messages: List[Dict],
    model: str = LLM_ENDPOINT,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    retries: int = 3
) -> Dict:
    """
    Call Databricks Foundation Model API with retry and exponential backoff.
    Returns the full response dict.
    """
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
    
    url = f"https://{workspace_url}/serving-endpoints/{model}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(retries):
        try:
            start = time.time()
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            latency_ms = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                result = resp.json()
                result["_latency_ms"] = latency_ms
                return result
            elif resp.status_code == 429:
                wait = (2 ** attempt) * 2
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise Exception(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    raise Exception("All retries exhausted")


# CELL 3 — Extraction Prompt (high-quality, medically-aware)
EXTRACTION_SYSTEM_PROMPT = """You are a medical facility data analyst specializing in Indian healthcare infrastructure. 
Your job is to extract structured medical capabilities from facility descriptions.

CRITICAL RULES:
1. NEVER infer capabilities not mentioned in the text. If unsure, use "uncertain".
2. Distinguish CAREFULLY between:
   - "has ICU" (claimed) vs "ICU is operational" (confirmed)
   - "visiting surgeon" (part_time/visiting) vs "full-time surgeon" (full_time)
   - "has ventilator" (claimed) vs evidence of actual use (confirmed)
3. For staff: "Dr. X visits on Tuesdays" = visiting. "24/7 doctor on duty" = full_time.
4. Always populate source_text with the EXACT substring from the input that supports each claim.
5. Return ONLY valid JSON matching the schema. No preamble, no explanation outside JSON.

STATUS VALUES:
- "confirmed": Explicitly stated as functional/operational
- "claimed": Listed but no functionality confirmation  
- "uncertain": Ambiguous or contradictory mention
- "not_present": Explicitly stated as absent OR no mention at all

STAFF TYPE VALUES:
- "full_time": On-site, always available
- "part_time": Regular but not full-time
- "visiting": Periodic visits only
- "on_call": Available by call
- "unknown": Not mentioned or unclear
"""

EXTRACTION_USER_TEMPLATE = """Extract medical capabilities from the following Indian healthcare facility record.

FACILITY TEXT:
{composite_text}

Return a JSON object with EXACTLY this structure:
{{
  "facility_id": "{facility_id}",
  "icu": {{
    "present": "<confirmed|claimed|uncertain|not_present>",
    "functional_status": "<functional|non_functional|partially_functional|unknown>",
    "bed_count": <integer or null>,
    "neonatal_icu": "<confirmed|claimed|uncertain|not_present>",
    "source_text": "<exact substring from input or null>"
  }},
  "ventilator": {{
    "present": "<confirmed|claimed|uncertain|not_present>",
    "count": <integer or null>,
    "reliability_note": "<string or null>",
    "source_text": "<exact substring from input or null>"
  }},
  "staff": {{
    "anesthesiologist": "<full_time|part_time|visiting|on_call|unknown>",
    "surgeon": "<full_time|part_time|visiting|on_call|unknown>",
    "general_physician": "<full_time|part_time|visiting|on_call|unknown>",
    "specialist_types": ["<specialty name>"],
    "total_doctor_count": <integer or null>,
    "source_text": "<exact substring from input or null>"
  }},
  "emergency": {{
    "emergency_care": "<confirmed|claimed|uncertain|not_present>",
    "is_24_7": <true|false>,
    "ambulance": "<confirmed|claimed|uncertain|not_present>",
    "trauma_capability": "<confirmed|claimed|uncertain|not_present>",
    "source_text": "<exact substring from input or null>"
  }},
  "surgery": {{
    "general_surgery": "<confirmed|claimed|uncertain|not_present>",
    "appendectomy": "<confirmed|claimed|uncertain|not_present>",
    "caesarean": "<confirmed|claimed|uncertain|not_present>",
    "orthopedic": "<confirmed|claimed|uncertain|not_present>",
    "cardiac": "<confirmed|claimed|uncertain|not_present>",
    "source_text": "<exact substring from input or null>"
  }},
  "dialysis": {{
    "present": "<confirmed|claimed|uncertain|not_present>",
    "machine_count": <integer or null>,
    "source_text": "<exact substring from input or null>"
  }},
  "specialties_extracted": ["<specialty name>"],
  "extraction_notes": "<any ambiguities or caveats as a string>",
  "raw_text_used": "<first 200 chars of input text>"
}}"""


# CELL 4 — Single-facility extraction function with MLflow tracing
def extract_facility(row: Row) -> Dict:
    """
    Extract structured capabilities for a single facility row.
    Wraps the LLM call in an MLflow span for full traceability.
    """
    facility_id = row["facility_id"]
    composite_text = row["composite_text"] or ""
    
    # Truncate to avoid token limits (keep most informative portion)
    truncated_text = composite_text[:3000] if len(composite_text) > 3000 else composite_text
    
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(
            composite_text=truncated_text,
            facility_id=facility_id
        )}
    ]
    
    with mlflow.start_span(
        name=f"extract_{facility_id[:8]}",
        span_type=SpanType.LLM,
        attributes={
            "facility_id": facility_id,
            "text_length": len(truncated_text),
            "model": LLM_ENDPOINT
        }
    ) as span:
        try:
            response = call_llm(messages)
            raw_content = response["choices"][0]["message"]["content"]
            tokens = response.get("usage", {})
            latency = response.get("_latency_ms", 0)
            
            # Parse JSON response
            # Strip markdown fences if model added them
            clean_content = raw_content.strip()
            if clean_content.startswith("```"):
                clean_content = clean_content.split("```")[1]
                if clean_content.startswith("json"):
                    clean_content = clean_content[4:]
                clean_content = clean_content.strip()
            
            extracted = json.loads(clean_content)
            extracted["_extraction_ok"] = True
            extracted["_tokens"] = tokens
            extracted["_latency_ms"] = latency
            
            span.set_attribute("tokens_prompt", tokens.get("prompt_tokens", 0))
            span.set_attribute("tokens_completion", tokens.get("completion_tokens", 0))
            span.set_attribute("extraction_ok", True)
            
            return extracted
            
        except json.JSONDecodeError as e:
            span.set_attribute("extraction_ok", False)
            span.set_attribute("error", str(e))
            return {
                "facility_id": facility_id,
                "_extraction_ok": False,
                "_error": f"JSON parse error: {str(e)}",
                "_raw_response": raw_content if 'raw_content' in dir() else ""
            }
        except Exception as e:
            span.set_attribute("extraction_ok", False)
            span.set_attribute("error", str(e))
            return {
                "facility_id": facility_id,
                "_extraction_ok": False,
                "_error": str(e)
            }


# CELL 5 — Batch extraction pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def extract_batch(rows: List[Row], max_workers: int = 5) -> List[Dict]:
    """
    Process a batch of rows concurrently with rate-limit awareness.
    max_workers=5 keeps us safely within free-tier API rate limits.
    """
    results = []
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_facility, row): row for row in rows}
        for future in as_completed(futures):
            try:
                result = future.result()
                with lock:
                    results.append(result)
            except Exception as e:
                row = futures[future]
                with lock:
                    results.append({
                        "facility_id": row["facility_id"],
                        "_extraction_ok": False,
                        "_error": str(e)
                    })
    return results


# CELL 6 — Run extraction on all 10k records (with checkpointing)
import math

BATCH_SIZE = 50        # Rows per batch
CHECKPOINT_EVERY = 10  # Save progress every N batches

df_bronze = spark.table(BRONZE_TABLE).select(
    "facility_id", "composite_text"
).where(F.col("composite_text").isNotNull())

# Collect to driver — for 10k rows this is manageable (~50MB)
all_rows = df_bronze.collect()
total = len(all_rows)
batches = [all_rows[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

print(f"Total records: {total}")
print(f"Batches: {len(batches)} × {BATCH_SIZE} rows")

all_results = []
failed_ids = []

with mlflow.start_run(run_name="facility_extraction_run"):
    mlflow.log_param("total_records", total)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("model", LLM_ENDPOINT)
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}...")
        batch_results = extract_batch(batch)
        
        ok = [r for r in batch_results if r.get("_extraction_ok", False)]
        fail = [r for r in batch_results if not r.get("_extraction_ok", True)]
        all_results.extend(ok)
        failed_ids.extend([r["facility_id"] for r in fail])
        
        # Checkpoint: save intermediate results every N batches
        if (i + 1) % CHECKPOINT_EVERY == 0:
            df_checkpoint = spark.createDataFrame(
                [Row(facility_id=r["facility_id"], extraction_json=json.dumps(r))
                 for r in all_results]
            )
            df_checkpoint.write.format("delta") \
                .mode("append") \
                .option("mergeSchema", "true") \
                .saveAsTable(SILVER_TABLE)
            print(f"  ✅ Checkpoint saved: {len(all_results)} records to {SILVER_TABLE}")
            all_results = []  # Clear buffer after checkpoint save
    
    # Final batch write
    if all_results:
        df_final = spark.createDataFrame(
            [Row(facility_id=r["facility_id"], extraction_json=json.dumps(r))
             for r in all_results]
        )
        df_final.write.format("delta") \
            .mode("append") \
            .option("mergeSchema", "true") \
            .saveAsTable(SILVER_TABLE)
    
    mlflow.log_metric("successful_extractions", spark.table(SILVER_TABLE).count())
    mlflow.log_metric("failed_extractions", len(failed_ids))
    
    if failed_ids:
        mlflow.log_text("\n".join(failed_ids), "failed_facility_ids.txt")

print(f"✅ Extraction complete. Silver table: {SILVER_TABLE}")
print(f"⚠️  Failed extractions: {len(failed_ids)}")
```

---

## 5. Notebook 03: Trust Scoring & Validation

**Notebook name:** `03_trust_scoring`

```python
# CELL 1 — Trust scoring engine

import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pyspark.sql import functions as F

# ─── Rule definitions ─────────────────────────────────────────────────────────

@dataclass
class TrustRule:
    rule_id: str
    severity: str  # LOW / MEDIUM / HIGH / CRITICAL
    description: str
    penalty: float  # Subtracted from base score 1.0

TRUST_RULES = [
    TrustRule(
        rule_id="SURGERY_NO_ANESTHESIOLOGIST",
        severity="CRITICAL",
        description="Surgery capability claimed but no anesthesiologist on staff",
        penalty=0.30
    ),
    TrustRule(
        rule_id="ICU_NON_FUNCTIONAL",
        severity="HIGH",
        description="ICU listed as present but marked non-functional",
        penalty=0.20
    ),
    TrustRule(
        rule_id="ICU_NO_BEDS",
        severity="MEDIUM",
        description="ICU presence claimed but no bed count provided",
        penalty=0.10
    ),
    TrustRule(
        rule_id="24_7_NO_STAFF",
        severity="HIGH",
        description="24/7 emergency claimed but no full-time doctor documented",
        penalty=0.20
    ),
    TrustRule(
        rule_id="CARDIAC_SURGERY_RURAL_CLINIC",
        severity="HIGH",
        description="Cardiac surgery claimed at a facility typed as 'clinic'",
        penalty=0.20
    ),
    TrustRule(
        rule_id="ZERO_DOCTORS_MANY_SPECIALTIES",
        severity="MEDIUM",
        description="Zero doctors listed but multiple specialties claimed",
        penalty=0.15
    ),
    TrustRule(
        rule_id="VENTILATOR_NO_ICU",
        severity="MEDIUM",
        description="Ventilators claimed but no ICU — unusual configuration",
        penalty=0.10
    ),
    TrustRule(
        rule_id="CAESAREAN_NO_ANESTHESIOLOGIST",
        severity="HIGH",
        description="C-section capability but no anesthesiologist",
        penalty=0.25
    ),
    TrustRule(
        rule_id="MISSING_CRITICAL_FIELDS",
        severity="LOW",
        description="Less than 3 extraction fields populated",
        penalty=0.10
    ),
    TrustRule(
        rule_id="EMPTY_DESCRIPTION",
        severity="LOW",
        description="No description or composite text available",
        penalty=0.05
    ),
]


def apply_rule_based_trust(
    extraction: Dict,
    facility_type: str = "unknown",
    number_doctors: int = None
) -> Tuple[float, List[Dict]]:
    """
    Apply deterministic trust rules to an extraction dict.
    Returns (trust_score, list_of_triggered_flags).
    """
    score = 1.0
    flags = []
    
    def flag(rule: TrustRule, evidence: str):
        nonlocal score
        score -= rule.penalty
        flags.append({
            "flag_type": rule.rule_id,
            "severity": rule.severity,
            "description": rule.description,
            "supporting_evidence": evidence
        })
    
    # --- Extract key fields from extraction dict ---
    icu = extraction.get("icu", {})
    ventilator = extraction.get("ventilator", {})
    staff = extraction.get("staff", {})
    emergency = extraction.get("emergency", {})
    surgery = extraction.get("surgery", {})
    dialysis = extraction.get("dialysis", {})
    
    icu_present = icu.get("present", "uncertain")
    icu_functional = icu.get("functional_status", "unknown")
    anesthesiologist = staff.get("anesthesiologist", "unknown")
    surgeon = staff.get("surgeon", "unknown")
    general_surgery = surgery.get("general_surgery", "uncertain")
    caesarean = surgery.get("caesarean", "uncertain")
    cardiac = surgery.get("cardiac", "uncertain")
    is_24_7 = emergency.get("is_24_7", False)
    gp = staff.get("general_physician", "unknown")
    vent_present = ventilator.get("present", "not_present")
    total_doctors = staff.get("total_doctor_count") or number_doctors
    specialist_types = staff.get("specialist_types", [])
    
    # Rule 1: Surgery without anesthesiologist
    surgery_claimed = general_surgery in ("confirmed", "claimed")
    caesarean_claimed = caesarean in ("confirmed", "claimed")
    if (surgery_claimed or caesarean_claimed) and anesthesiologist == "unknown":
        flag(TRUST_RULES[0], 
             f"surgery={general_surgery}, caesarean={caesarean}, anesthesiologist={anesthesiologist}")
    
    # Rule 2: ICU listed but non-functional
    if icu_present in ("confirmed", "claimed") and icu_functional == "non_functional":
        flag(TRUST_RULES[1],
             f"icu.present={icu_present}, icu.functional_status={icu_functional}")
    
    # Rule 3: ICU with no bed count
    if icu_present == "confirmed" and icu.get("bed_count") is None:
        flag(TRUST_RULES[2], "ICU confirmed but bed_count=null")
    
    # Rule 4: 24/7 emergency without full-time staff
    if is_24_7 and gp not in ("full_time", "on_call") and surgeon not in ("full_time", "on_call"):
        flag(TRUST_RULES[3],
             f"is_24_7=true but GP={gp}, surgeon={surgeon}")
    
    # Rule 5: Cardiac surgery at a clinic
    if cardiac in ("confirmed", "claimed") and "clinic" in facility_type.lower():
        flag(TRUST_RULES[4],
             f"cardiac={cardiac}, facility_type={facility_type}")
    
    # Rule 6: Zero doctors with many specialties
    if total_doctors is not None and total_doctors == 0 and len(specialist_types) > 3:
        flag(TRUST_RULES[5],
             f"total_doctors=0, specialist_types_count={len(specialist_types)}")
    
    # Rule 7: Ventilator without ICU
    if vent_present in ("confirmed", "claimed") and icu_present == "not_present":
        flag(TRUST_RULES[6],
             f"ventilator={vent_present}, icu={icu_present}")
    
    # Rule 8: C-section without anesthesiologist
    if caesarean in ("confirmed", "claimed") and anesthesiologist == "unknown":
        flag(TRUST_RULES[7],
             f"caesarean={caesarean}, anesthesiologist={anesthesiologist}")
    
    # Count populated fields
    populated = sum([
        icu_present != "uncertain",
        vent_present != "uncertain",
        anesthesiologist != "unknown",
        emergency.get("emergency_care", "uncertain") != "uncertain",
        general_surgery != "uncertain",
        dialysis.get("present", "uncertain") != "uncertain"
    ])
    
    if populated < 3:
        flag(TRUST_RULES[8], f"Only {populated}/6 core fields extracted")
    
    if not extraction.get("raw_text_used", "").strip():
        flag(TRUST_RULES[9], "raw_text_used is empty")
    
    # Floor at 0.05 — never zero unless completely empty
    score = max(0.05, min(1.0, score))
    return round(score, 3), flags


# CELL 2 — LLM-based contradiction detection
VALIDATOR_SYSTEM_PROMPT = """You are a medical quality assurance agent reviewing healthcare facility extractions.
Your task: identify logical contradictions and data quality issues that rule-based checks might miss.

Focus on:
1. Medical impossibilities (e.g., dialysis unit in a single-doctor dental clinic)
2. Equipment–staff mismatches (e.g., CT scanner but no radiologist even mentioned)
3. Capacity contradictions (e.g., claims 500 beds but description says "small rural clinic")
4. Geographic implausibility (e.g., claims multi-specialty but in remote tribal area with 0 doctors)
5. Temporal inconsistencies (e.g., "newly established" but claims 20 years of surgical history)

Return ONLY valid JSON:
{
  "has_contradictions": <true|false>,
  "contradiction_flags": [
    {
      "flag_type": "DESCRIPTIVE_FLAG_NAME",
      "severity": "<LOW|MEDIUM|HIGH|CRITICAL>",
      "description": "<what the contradiction is>",
      "supporting_evidence": "<quote from extraction that shows the issue>"
    }
  ],
  "validator_notes": "<overall assessment in 1-2 sentences>",
  "recommend_reextraction": <true|false>
}"""

def validate_with_llm(extraction: Dict, facility_meta: Dict) -> Dict:
    """
    Use LLM to find contradictions rule-based scoring might miss.
    Returns validator response dict.
    """
    context = {
        "facility_name": facility_meta.get("name", "Unknown"),
        "facility_type": facility_meta.get("facilityTypeId", "unknown"),
        "city": facility_meta.get("address_city", ""),
        "state": facility_meta.get("address_stateOrRegion", ""),
        "number_doctors": facility_meta.get("numberDoctors", "unknown"),
        "extraction_summary": {
            "icu": extraction.get("icu", {}),
            "surgery": extraction.get("surgery", {}),
            "staff": extraction.get("staff", {}),
            "emergency": extraction.get("emergency", {}),
        }
    }
    
    messages = [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Review this facility extraction:\n\n{json.dumps(context, indent=2)}"}
    ]
    
    try:
        response = call_llm(messages, max_tokens=512)
        raw = response["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {
            "has_contradictions": False,
            "contradiction_flags": [],
            "validator_notes": f"Validation failed: {str(e)}",
            "recommend_reextraction": False
        }


# CELL 3 — Confidence scoring model
def compute_confidence(
    extraction: Dict,
    trust_score: float,
    trust_flags: List[Dict],
    composite_text_length: int
) -> Dict:
    """
    Probabilistic confidence model based on:
    - completeness: fraction of key fields extracted
    - consistency: absence of contradictions
    - reliability: source signal strength
    Returns scores with confidence intervals.
    """
    # Completeness: count non-uncertain/non-unknown fields out of 6 key dimensions
    key_fields = [
        extraction.get("icu", {}).get("present", "uncertain") not in ("uncertain",),
        extraction.get("ventilator", {}).get("present", "uncertain") not in ("uncertain",),
        extraction.get("staff", {}).get("anesthesiologist", "unknown") not in ("unknown",),
        extraction.get("emergency", {}).get("emergency_care", "uncertain") not in ("uncertain",),
        extraction.get("surgery", {}).get("general_surgery", "uncertain") not in ("uncertain",),
        extraction.get("dialysis", {}).get("present", "uncertain") not in ("uncertain",),
    ]
    completeness = sum(key_fields) / len(key_fields)
    
    # Consistency: penalize for each HIGH/CRITICAL flag
    high_flags = sum(1 for f in trust_flags if f["severity"] in ("HIGH", "CRITICAL"))
    med_flags = sum(1 for f in trust_flags if f["severity"] == "MEDIUM")
    consistency = max(0.0, 1.0 - (high_flags * 0.25) - (med_flags * 0.10))
    
    # Reliability: based on text richness (more text = more signal)
    if composite_text_length > 1000:
        reliability = 0.85
    elif composite_text_length > 400:
        reliability = 0.65
    elif composite_text_length > 100:
        reliability = 0.45
    else:
        reliability = 0.20
    
    # Weighted overall
    overall = (completeness * 0.40) + (consistency * 0.35) + (reliability * 0.25)
    overall = round(overall, 3)
    
    # Bootstrap-inspired CI: simulate variance from flag count uncertainty
    flag_noise = 0.05 + (len(trust_flags) * 0.02)
    ci_low = max(0.0, round(overall - flag_noise, 3))
    ci_high = min(1.0, round(overall + flag_noise, 3))
    
    return {
        "completeness": round(completeness, 3),
        "consistency": round(consistency, 3),
        "reliability": round(reliability, 3),
        "overall": overall,
        "confidence_interval_low": ci_low,
        "confidence_interval_high": ci_high
    }


# CELL 4 — Assemble Gold table records
from pyspark.sql.types import *

silver_df = spark.table(SILVER_TABLE)
bronze_df = spark.table(BRONZE_TABLE).select(
    "facility_id", "name", "address_city", "address_stateOrRegion",
    "address_zipOrPostcode", "latitude", "longitude",
    "facilityTypeId", "operatorTypeId", "numberDoctors",
    "composite_text", "composite_text_length"
)

# Join silver + bronze
joined = silver_df.join(bronze_df, "facility_id", "left")
rows = joined.collect()

gold_records = []
for row in rows:
    try:
        extraction = json.loads(row["extraction_json"])
        facility_meta = {
            "name": row["name"],
            "facilityTypeId": row["facilityTypeId"],
            "address_city": row["address_city"],
            "address_stateOrRegion": row["address_stateOrRegion"],
            "numberDoctors": row["numberDoctors"]
        }
        
        # Rule-based trust scoring
        trust_score, trust_flags = apply_rule_based_trust(
            extraction,
            facility_type=row["facilityTypeId"] or "",
            number_doctors=int(row["numberDoctors"]) if row["numberDoctors"] else None
        )
        
        # Confidence scoring
        confidence = compute_confidence(
            extraction,
            trust_score,
            trust_flags,
            int(row["composite_text_length"] or 0)
        )
        
        # Build embedding text (searchable, rich concatenation)
        specialties = extraction.get("specialties_extracted", [])
        icu_status = extraction.get("icu", {}).get("present", "unknown")
        emergency_status = extraction.get("emergency", {}).get("emergency_care", "unknown")
        
        embedding_text = (
            f"Facility: {row['name']}. "
            f"Location: {row['address_city']}, {row['address_stateOrRegion']}, PIN {row['address_zipOrPostcode']}. "
            f"Type: {row['facilityTypeId']}. "
            f"Specialties: {', '.join(specialties)}. "
            f"ICU: {icu_status}. Emergency: {emergency_status}. "
            f"Trust Score: {trust_score}. "
            f"{extraction.get('raw_text_used', '')[:300]}"
        )
        
        gold_records.append({
            "facility_id": row["facility_id"],
            "name": row["name"] or "",
            "address_city": row["address_city"],
            "address_stateOrRegion": row["address_stateOrRegion"],
            "address_zipOrPostcode": row["address_zipOrPostcode"],
            "latitude": float(row["latitude"]) if row["latitude"] else None,
            "longitude": float(row["longitude"]) if row["longitude"] else None,
            "facilityTypeId": row["facilityTypeId"],
            "operatorTypeId": row["operatorTypeId"],
            "extraction_json": json.dumps(extraction),
            "trust_score": trust_score,
            "trust_flags_json": json.dumps(trust_flags),
            "confidence_json": json.dumps(confidence),
            "correction_iterations": 0,
            "embedding_text": embedding_text[:2000],
            "extraction_version": "1.0"
        })
    except Exception as e:
        print(f"Error processing {row['facility_id']}: {e}")

# Write Gold table
df_gold = spark.createDataFrame(gold_records)
df_gold.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(GOLD_TABLE)

print(f"✅ Gold table written: {df_gold.count()} rows → {GOLD_TABLE}")
print(f"   Average trust score: {df_gold.agg({'trust_score': 'avg'}).collect()[0][0]:.3f}")

# CELL 5 — Optimize Gold table for query performance
spark.sql(f"OPTIMIZE {GOLD_TABLE} ZORDER BY (address_stateOrRegion, address_zipOrPostcode)")
print("✅ OPTIMIZE + ZORDER complete")
```

---

## 6. Notebook 04: Self-Correction Agent Loop

**Notebook name:** `04_self_correction`

```python
# CELL 1 — Self-correction loop: Extractor → Validator → Corrector

CORRECTOR_SYSTEM_PROMPT = """You are a medical data correction agent. You have received:
1. An original extraction from a healthcare facility description
2. A validator's report listing specific contradictions and concerns
3. The original facility text

Your task: produce a CORRECTED extraction that resolves the contradictions.

Rules:
- Do NOT invent information not present in the original text
- If a contradiction cannot be resolved from the text, set the field to "uncertain"/"unknown"
- Always update source_text to point to the actual evidence
- Be conservative: it's better to say "uncertain" than to guess

Return ONLY corrected JSON in the same schema as the original extraction."""

def run_self_correction_loop(
    facility_id: str,
    composite_text: str,
    initial_extraction: Dict,
    facility_meta: Dict,
    max_iterations: int = 2
) -> Tuple[Dict, int, List[Dict]]:
    """
    Runs the Extractor → Validator → Corrector loop.
    Returns (final_extraction, iterations_run, all_validator_reports).
    """
    current_extraction = initial_extraction.copy()
    all_validator_reports = []
    iterations = 0
    
    for iteration in range(max_iterations):
        # Step 1: Validate current extraction
        with mlflow.start_span(
            name=f"validate_iter_{iteration}_{facility_id[:8]}",
            span_type=SpanType.LLM,
            attributes={"iteration": iteration, "facility_id": facility_id}
        ):
            validator_report = validate_with_llm(current_extraction, facility_meta)
            all_validator_reports.append(validator_report)
        
        # Step 2: Check if correction is needed
        if not validator_report.get("recommend_reextraction", False):
            print(f"  ✅ {facility_id[:12]}: Passed validation at iteration {iteration}")
            break
        
        # Step 3: Run corrector
        print(f"  🔄 {facility_id[:12]}: Correction needed (iter {iteration+1})")
        
        correction_prompt = f"""Original facility text:
{composite_text[:2000]}

Validator report:
{json.dumps(validator_report, indent=2)}

Current extraction (to be corrected):
{json.dumps(current_extraction, indent=2)}

Produce the corrected extraction JSON:"""
        
        messages = [
            {"role": "system", "content": CORRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": correction_prompt}
        ]
        
        with mlflow.start_span(
            name=f"correct_iter_{iteration}_{facility_id[:8]}",
            span_type=SpanType.LLM
        ):
            try:
                response = call_llm(messages, max_tokens=2048)
                raw = response["choices"][0]["message"]["content"].strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                corrected = json.loads(raw.strip())
                corrected["_correction_iteration"] = iteration + 1
                current_extraction = corrected
                iterations = iteration + 1
            except Exception as e:
                print(f"  ⚠️  Correction parse failed: {e}. Keeping previous extraction.")
                break
    
    return current_extraction, iterations, all_validator_reports


# CELL 2 — Run self-correction on HIGH-RISK records only
# Strategy: full correction loop only for low-trust or flagged records
# (saves compute on free tier — don't run full loop on all 10k)

CORRECTION_THRESHOLD = 0.65  # Run loop only if trust_score < this

gold_df = spark.table(GOLD_TABLE)
bronze_df = spark.table(BRONZE_TABLE).select("facility_id", "composite_text")

# Filter to records needing correction
needs_correction = gold_df.filter(
    F.col("trust_score") < CORRECTION_THRESHOLD
).join(bronze_df, "facility_id", "left") \
 .select("facility_id", "name", "facilityTypeId", "address_city",
         "address_stateOrRegion", "extraction_json", "composite_text") \
 .collect()

print(f"Records requiring self-correction: {len(needs_correction)}")

corrected_updates = []

with mlflow.start_run(run_name="self_correction_loop"):
    for row in needs_correction[:500]:  # Cap at 500 for free tier
        try:
            extraction = json.loads(row["extraction_json"])
            facility_meta = {
                "name": row["name"],
                "facilityTypeId": row["facilityTypeId"],
                "address_city": row["address_city"],
                "address_stateOrRegion": row["address_stateOrRegion"]
            }
            
            final_extraction, iterations, reports = run_self_correction_loop(
                facility_id=row["facility_id"],
                composite_text=row["composite_text"] or "",
                initial_extraction=extraction,
                facility_meta=facility_meta,
                max_iterations=2
            )
            
            if iterations > 0:
                # Re-score after correction
                new_trust, new_flags = apply_rule_based_trust(
                    final_extraction,
                    facility_type=row["facilityTypeId"] or ""
                )
                
                corrected_updates.append({
                    "facility_id": row["facility_id"],
                    "extraction_json": json.dumps(final_extraction),
                    "trust_score": new_trust,
                    "trust_flags_json": json.dumps(new_flags),
                    "correction_iterations": iterations
                })
        except Exception as e:
            print(f"  ⚠️  Failed self-correction for {row['facility_id']}: {e}")

# CELL 3 — Merge corrections back into Gold table
if corrected_updates:
    df_corrections = spark.createDataFrame(corrected_updates)
    
    from delta.tables import DeltaTable
    gold_delta = DeltaTable.forName(spark, GOLD_TABLE)
    
    gold_delta.alias("gold").merge(
        df_corrections.alias("corrections"),
        "gold.facility_id = corrections.facility_id"
    ).whenMatchedUpdate(set={
        "extraction_json": "corrections.extraction_json",
        "trust_score": "corrections.trust_score",
        "trust_flags_json": "corrections.trust_flags_json",
        "correction_iterations": "corrections.correction_iterations"
    }).execute()
    
    print(f"✅ Merged {len(corrected_updates)} corrected records into Gold table")
```

---

## 7. Notebook 05: Vector Search Setup

**Notebook name:** `05_vector_search`

```python
# CELL 1 — Mosaic AI Vector Search setup
# Databricks Free Edition includes Mosaic AI Vector Search (serverless)
# Documentation: https://docs.databricks.com/en/generative-ai/vector-search.html

from databricks.vector_search.client import VectorSearchClient

VS_ENDPOINT_NAME = "healthcare_vs_endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.facilities_vs_index"
EMBEDDING_MODEL = "databricks-gte-large-en"  # Available on free tier

vsc = VectorSearchClient()

# CELL 2 — Create Vector Search endpoint (one-time setup)
# Check if endpoint exists first
try:
    endpoint_info = vsc.get_endpoint(VS_ENDPOINT_NAME)
    print(f"✅ Endpoint already exists: {VS_ENDPOINT_NAME}")
except Exception:
    print(f"Creating endpoint: {VS_ENDPOINT_NAME}")
    vsc.create_endpoint(
        name=VS_ENDPOINT_NAME,
        endpoint_type="STANDARD"  # Free tier uses STANDARD
    )
    # Wait for endpoint to be ready
    import time
    for _ in range(30):
        status = vsc.get_endpoint(VS_ENDPOINT_NAME)["endpoint_status"]["state"]
        if status == "ONLINE":
            print(f"✅ Endpoint ready")
            break
        print(f"  Waiting... status={status}")
        time.sleep(20)

# CELL 3 — Enable Change Data Feed on Gold table (required for sync)
spark.sql(f"ALTER TABLE {GOLD_TABLE} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

# CELL 4 — Create Delta Sync Vector Search index
# This index auto-syncs when Gold table is updated
try:
    index = vsc.get_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME
    )
    print(f"✅ Index already exists: {VS_INDEX_NAME}")
except Exception:
    print(f"Creating vector search index: {VS_INDEX_NAME}")
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        source_table_name=GOLD_TABLE,
        pipeline_type="TRIGGERED",          # TRIGGERED = manual sync (saves compute)
        primary_key="facility_id",
        embedding_source_column="embedding_text",
        embedding_model_endpoint_name=EMBEDDING_MODEL
    )
    print(f"✅ Index created: {VS_INDEX_NAME}")

# CELL 5 — Trigger initial sync and wait
index.sync()
print("⏳ Syncing index...")

for _ in range(60):
    idx_info = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    status = idx_info.get("status", {}).get("ready", False)
    num_indexed = idx_info.get("status", {}).get("indexed_row_count", 0)
    print(f"  Status: ready={status}, indexed_rows={num_indexed}")
    if status:
        print("✅ Index sync complete")
        break
    time.sleep(30)


# CELL 6 — Vector search query function
def vector_search(
    query_text: str,
    top_k: int = 20,
    filters: Dict = None
) -> List[Dict]:
    """
    Perform semantic vector search over Gold table.
    filters: dict with keys like {"address_stateOrRegion": "Bihar", "trust_score": ">0.5"}
    Returns list of facility dicts with score.
    """
    vs_index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    
    # Build filter string for Databricks VS
    filter_dict = {}
    if filters:
        for k, v in filters.items():
            filter_dict[k] = v
    
    results = vs_index.similarity_search(
        query_text=query_text,
        columns=[
            "facility_id", "name", "address_city", "address_stateOrRegion",
            "address_zipOrPostcode", "latitude", "longitude",
            "facilityTypeId", "trust_score", "confidence_json",
            "extraction_json", "embedding_text"
        ],
        filters=filter_dict if filter_dict else None,
        num_results=top_k
    )
    
    return results.get("result", {}).get("data_array", [])


# CELL 7 — Test the vector search
test_results = vector_search(
    query_text="emergency surgery ICU ventilator rural Bihar",
    top_k=5,
    filters={"address_stateOrRegion": "Bihar"}
)
print(f"Test search returned {len(test_results)} results")
for r in test_results[:3]:
    print(f"  - {r[1]} ({r[3]}) | Trust: {r[7]:.2f}")
```

---

## 8. Notebook 06: Multi-Attribute Reasoning Engine

**Notebook name:** `06_reasoning_engine`

```python
# CELL 1 — Reasoning engine: Vector → Filter → LLM Rank

REASONING_SYSTEM_PROMPT = """You are a clinical decision support agent for Indian healthcare.
You help users find the most appropriate medical facility for their specific need.

You will receive:
1. A user's medical query
2. A list of candidate facilities (pre-filtered by location and semantic search)
3. Each facility's extracted capabilities, trust score, and confidence level

Your task:
1. Rank the candidates from most to least suitable for this specific query
2. For each top candidate, explain WHY it's suitable or not suitable
3. Flag any trust/confidence concerns the user should be aware of
4. If no facility adequately meets the query, say so clearly

CRITICAL: Only recommend facilities whose trust_score >= 0.5.
CRITICAL: Always cite the specific source_text from extractions when making claims.
CRITICAL: Warn if a needed capability has AvailabilityStatus = "claimed" vs "confirmed".

Return JSON:
{
  "query_interpretation": "<what the query is really asking for>",
  "ranked_results": [
    {
      "rank": 1,
      "facility_id": "<id>",
      "facility_name": "<name>",
      "suitability_score": <0.0-1.0>,
      "reasoning": "<why this facility fits or doesn't>",
      "matched_capabilities": ["<list of matching capabilities>"],
      "warnings": ["<any concerns about this facility>"],
      "citations": ["<exact source_text snippets that support the recommendation>"]
    }
  ],
  "recommendation_summary": "<1-2 sentence summary for a non-technical user>",
  "uncertainty_note": "<any caveats about data quality or confidence>"
}"""

def query_facilities(
    user_query: str,
    state_filter: str = None,
    city_filter: str = None,
    facility_type_filter: str = None,
    min_trust_score: float = 0.4,
    top_k_vector: int = 20,
    top_k_final: int = 5
) -> Dict:
    """
    Full multi-attribute reasoning pipeline:
    1. Vector search (semantic retrieval)
    2. Structured SQL filter (hard constraints)
    3. LLM reasoning + ranking
    4. MLflow trace
    
    Returns ranked results with explanations.
    """
    with mlflow.start_run(run_name=f"query_{user_query[:30]}"):
        
        # ── Step 1: Vector retrieval ──────────────────────────────────────────
        vs_filters = {}
        if state_filter:
            vs_filters["address_stateOrRegion"] = state_filter
        
        with mlflow.start_span(name="vector_retrieval", span_type=SpanType.RETRIEVER) as span:
            candidates_raw = vector_search(
                query_text=user_query,
                top_k=top_k_vector,
                filters=vs_filters if vs_filters else None
            )
            span.set_attribute("candidates_retrieved", len(candidates_raw))
        
        if not candidates_raw:
            return {"error": "No candidates found. Try broadening your search criteria."}
        
        # Convert to list of dicts (VS returns list of lists)
        cols = ["facility_id", "name", "address_city", "address_stateOrRegion",
                "address_zipOrPostcode", "latitude", "longitude",
                "facilityTypeId", "trust_score", "confidence_json",
                "extraction_json", "embedding_text"]
        candidates = [dict(zip(cols, row)) for row in candidates_raw]
        
        # ── Step 2: Structured SQL filter (non-LLM, deterministic) ───────────
        with mlflow.start_span(name="structured_filter") as span:
            candidate_ids = [c["facility_id"] for c in candidates]
            id_list = "', '".join(candidate_ids)
            
            filter_clauses = [f"facility_id IN ('{id_list}')"]
            filter_clauses.append(f"trust_score >= {min_trust_score}")
            if city_filter:
                filter_clauses.append(f"address_city = '{city_filter}'")
            if facility_type_filter:
                filter_clauses.append(f"facilityTypeId = '{facility_type_filter}'")
            
            where_clause = " AND ".join(filter_clauses)
            
            filtered_df = spark.sql(f"""
                SELECT facility_id, trust_score, extraction_json, confidence_json
                FROM {GOLD_TABLE}
                WHERE {where_clause}
            """).collect()
            
            filtered_ids = {row["facility_id"]: row for row in filtered_df}
            candidates = [c for c in candidates if c["facility_id"] in filtered_ids]
            
            # Merge in fresh trust/extraction data from Gold table
            for c in candidates:
                gold_row = filtered_ids[c["facility_id"]]
                c["trust_score"] = float(gold_row["trust_score"])
                c["extraction_json"] = gold_row["extraction_json"]
                c["confidence_json"] = gold_row["confidence_json"]
            
            span.set_attribute("after_filter_count", len(candidates))
        
        if not candidates:
            return {"error": f"No facilities meet the criteria (trust >= {min_trust_score}, state={state_filter})."}
        
        # ── Step 3: Build context for LLM ranker ─────────────────────────────
        facility_summaries = []
        for c in candidates[:top_k_vector]:
            extraction = json.loads(c.get("extraction_json", "{}"))
            confidence = json.loads(c.get("confidence_json", "{}"))
            
            summary = {
                "facility_id": c["facility_id"],
                "name": c["name"],
                "location": f"{c['address_city']}, {c['address_stateOrRegion']} - {c['address_zipOrPostcode']}",
                "facility_type": c["facilityTypeId"],
                "trust_score": round(c["trust_score"], 3),
                "confidence": confidence.get("overall", 0),
                "capabilities": {
                    "icu": extraction.get("icu", {}).get("present", "unknown"),
                    "icu_functional": extraction.get("icu", {}).get("functional_status", "unknown"),
                    "surgery": extraction.get("surgery", {}),
                    "emergency_24_7": extraction.get("emergency", {}).get("is_24_7", False),
                    "anesthesiologist": extraction.get("staff", {}).get("anesthesiologist", "unknown"),
                    "surgeon_type": extraction.get("staff", {}).get("surgeon", "unknown"),
                    "dialysis": extraction.get("dialysis", {}).get("present", "unknown"),
                    "specialties": extraction.get("specialties_extracted", [])[:10]
                },
                "key_source_texts": [
                    extraction.get("surgery", {}).get("source_text"),
                    extraction.get("staff", {}).get("source_text"),
                    extraction.get("emergency", {}).get("source_text")
                ]
            }
            facility_summaries.append(summary)
        
        # ── Step 4: LLM reasoning + ranking ──────────────────────────────────
        reasoning_prompt = f"""User Query: "{user_query}"

Candidate Facilities ({len(facility_summaries)} total):
{json.dumps(facility_summaries, indent=2)}

Rank and evaluate these candidates for the user's specific medical need.
Return top {top_k_final} with detailed reasoning."""
        
        messages = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": reasoning_prompt}
        ]
        
        with mlflow.start_span(name="llm_reasoning", span_type=SpanType.LLM) as span:
            response = call_llm(messages, max_tokens=2000, temperature=0.1)
            raw = response["choices"][0]["message"]["content"].strip()
            tokens = response.get("usage", {})
            span.set_attribute("tokens_used", tokens.get("total_tokens", 0))
            
            # Parse response
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            reasoning_result = json.loads(raw.strip())
        
        # Log everything to MLflow
        mlflow.log_param("user_query", user_query)
        mlflow.log_param("state_filter", state_filter)
        mlflow.log_metric("candidates_retrieved", len(candidates_raw))
        mlflow.log_metric("candidates_after_filter", len(candidates))
        mlflow.log_metric("top_result_trust", 
                          reasoning_result["ranked_results"][0].get("suitability_score", 0)
                          if reasoning_result.get("ranked_results") else 0)
        
        return reasoning_result


# CELL 2 — Test the reasoning engine
test_result = query_facilities(
    user_query="Find a facility that can perform emergency appendectomy with part-time doctors",
    state_filter="Bihar",
    min_trust_score=0.5,
    top_k_final=3
)
print(json.dumps(test_result, indent=2))
```

---

## 9. Notebook 07: Medical Desert Detection

**Notebook name:** `07_medical_deserts`

```python
# CELL 1 — Medical desert detection by PIN code

from pyspark.sql import functions as F, Window
from pyspark.sql.types import *

# ── Step 1: Parse extraction fields into queryable columns ────────────────────
gold_df = spark.table(GOLD_TABLE)

# UDFs to parse JSON extraction fields
@F.udf(returnType=StringType())
def get_icu_status(extraction_json):
    try:
        return json.loads(extraction_json).get("icu", {}).get("present", "uncertain")
    except:
        return "uncertain"

@F.udf(returnType=StringType())
def get_dialysis_status(extraction_json):
    try:
        return json.loads(extraction_json).get("dialysis", {}).get("present", "uncertain")
    except:
        return "uncertain"

@F.udf(returnType=StringType())
def get_emergency_status(extraction_json):
    try:
        return json.loads(extraction_json).get("emergency", {}).get("emergency_care", "uncertain")
    except:
        return "uncertain"

@F.udf(returnType=StringType())
def get_surgery_status(extraction_json):
    try:
        return json.loads(extraction_json).get("surgery", {}).get("general_surgery", "uncertain")
    except:
        return "uncertain"

@F.udf(returnType=BooleanType())
def get_icu_functional(extraction_json):
    try:
        ext = json.loads(extraction_json)
        icu = ext.get("icu", {})
        return (icu.get("present") in ("confirmed", "claimed") and 
                icu.get("functional_status") not in ("non_functional",))
    except:
        return False

enriched_gold = gold_df.withColumn("icu_status", get_icu_status("extraction_json")) \
    .withColumn("dialysis_status", get_dialysis_status("extraction_json")) \
    .withColumn("emergency_status", get_emergency_status("extraction_json")) \
    .withColumn("surgery_status", get_surgery_status("extraction_json")) \
    .withColumn("icu_functional", get_icu_functional("extraction_json")) \
    .withColumn("has_icu", F.col("icu_status").isin("confirmed", "claimed").cast("int")) \
    .withColumn("has_dialysis", F.col("dialysis_status").isin("confirmed", "claimed").cast("int")) \
    .withColumn("has_emergency", F.col("emergency_status").isin("confirmed", "claimed").cast("int")) \
    .withColumn("has_surgery", F.col("surgery_status").isin("confirmed", "claimed").cast("int"))


# CELL 2 — Aggregate by PIN code
desert_by_pin = enriched_gold.groupBy(
    "address_zipOrPostcode",
    "address_stateOrRegion"
).agg(
    F.count("*").alias("facility_count"),
    F.avg("trust_score").alias("avg_trust_score"),
    
    # Coverage = fraction of facilities in PIN with this capability
    F.round(F.avg("has_icu"), 3).alias("icu_coverage"),
    F.round(F.avg("has_dialysis"), 3).alias("dialysis_coverage"),
    F.round(F.avg("has_emergency"), 3).alias("emergency_coverage"),
    F.round(F.avg("has_surgery"), 3).alias("surgery_coverage"),
    
    # Geographic centroid
    F.avg("latitude").alias("centroid_lat"),
    F.avg("longitude").alias("centroid_lon")
).where(
    F.col("address_zipOrPostcode").isNotNull()
)

# CELL 3 — Compute desert risk score
# Risk = weighted sum of capability gaps
# Higher = worse (closer to complete desert)
ICU_WEIGHT = 0.35
DIALYSIS_WEIGHT = 0.20
EMERGENCY_WEIGHT = 0.30
SURGERY_WEIGHT = 0.15
HIGH_RISK_THRESHOLD = 0.70

desert_scored = desert_by_pin.withColumn(
    "desert_risk_score",
    F.round(
        (F.lit(ICU_WEIGHT) * (F.lit(1.0) - F.col("icu_coverage"))) +
        (F.lit(DIALYSIS_WEIGHT) * (F.lit(1.0) - F.col("dialysis_coverage"))) +
        (F.lit(EMERGENCY_WEIGHT) * (F.lit(1.0) - F.col("emergency_coverage"))) +
        (F.lit(SURGERY_WEIGHT) * (F.lit(1.0) - F.col("surgery_coverage"))),
        3
    )
).withColumn(
    "is_high_risk",
    F.col("desert_risk_score") >= HIGH_RISK_THRESHOLD
)

# CELL 4 — Flag desert categories
@F.udf(returnType=ArrayType(StringType()))
def get_desert_categories(icu_cov, dialysis_cov, emergency_cov, surgery_cov):
    categories = []
    DESERT_THRESHOLD = 0.10  # <10% facilities have capability = desert
    if icu_cov is not None and icu_cov < DESERT_THRESHOLD:
        categories.append("ICU_DESERT")
    if dialysis_cov is not None and dialysis_cov < DESERT_THRESHOLD:
        categories.append("DIALYSIS_DESERT")
    if emergency_cov is not None and emergency_cov < DESERT_THRESHOLD:
        categories.append("EMERGENCY_DESERT")
    if surgery_cov is not None and surgery_cov < DESERT_THRESHOLD:
        categories.append("SURGERY_DESERT")
    if not categories:
        categories.append("ADEQUATE_COVERAGE")
    return categories

desert_final = desert_scored.withColumn(
    "desert_categories",
    get_desert_categories("icu_coverage", "dialysis_coverage", 
                          "emergency_coverage", "surgery_coverage")
)

# CELL 5 — Write Desert table
desert_final.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(DESERT_TABLE)

print(f"✅ Desert table written: {desert_final.count()} PIN codes analyzed")

# CELL 6 — Summary statistics
high_risk = desert_final.filter(F.col("is_high_risk"))
print(f"\n=== MEDICAL DESERT REPORT ===")
print(f"Total PIN codes analyzed: {desert_final.count()}")
print(f"High-risk deserts: {high_risk.count()}")
print(f"\nTop 10 highest-risk states:")
display(
    desert_final.groupBy("address_stateOrRegion")
    .agg(
        F.count("*").alias("pin_codes"),
        F.sum(F.col("is_high_risk").cast("int")).alias("high_risk_pins"),
        F.avg("desert_risk_score").alias("avg_desert_risk")
    )
    .orderBy(F.desc("avg_desert_risk"))
    .limit(10)
)

# CELL 7 — Geo-ready export for map visualization
geo_export = desert_final.select(
    F.col("address_zipOrPostcode").alias("pin_code"),
    F.col("address_stateOrRegion").alias("state"),
    "facility_count",
    "icu_coverage", "dialysis_coverage", "emergency_coverage", "surgery_coverage",
    "desert_risk_score", "desert_categories", "is_high_risk",
    "avg_trust_score", "centroid_lat", "centroid_lon"
).where(
    F.col("centroid_lat").isNotNull() & F.col("centroid_lon").isNotNull()
)

# Export as JSON for map consumption
geo_json_path = "/dbfs/FileStore/desert_map_export.json"
geo_export.toPandas().to_json(geo_json_path, orient="records")
print(f"✅ Geo-ready JSON exported: {geo_json_path}")
```

---

## 10. Notebook 08: API Serving Layer

**Notebook name:** `08_api_serving`

```python
# CELL 1 — API layer: clean Python functions usable as Databricks Model endpoints
# or called from a FastAPI app deployed on Databricks Apps

import json
from typing import Optional, List, Dict, Any
from pyspark.sql import functions as F

# ════════════════════════════════════════════════════════════════════════
# API FUNCTION 1: Query Facilities
# ════════════════════════════════════════════════════════════════════════

def api_query_facilities(
    query: str,
    state: Optional[str] = None,
    city: Optional[str] = None,
    facility_type: Optional[str] = None,
    min_trust_score: float = 0.4,
    top_k: int = 5,
    include_chain_of_thought: bool = True
) -> Dict:
    """
    PRIMARY QUERY ENDPOINT
    
    Example call:
        api_query_facilities(
            query="emergency appendectomy with part-time doctors",
            state="Bihar",
            min_trust_score=0.5,
            top_k=3
        )
    
    Returns:
        {
            "query": ...,
            "ranked_results": [...],
            "recommendation_summary": ...,
            "uncertainty_note": ...,
            "chain_of_thought": {...}  # if include_chain_of_thought=True
        }
    """
    result = query_facilities(
        user_query=query,
        state_filter=state,
        city_filter=city,
        facility_type_filter=facility_type,
        min_trust_score=min_trust_score,
        top_k_final=top_k
    )
    
    if not include_chain_of_thought:
        result.pop("_debug", None)
    
    result["api_version"] = "1.0"
    result["query_received"] = query
    return result


# ════════════════════════════════════════════════════════════════════════
# API FUNCTION 2: Get Facility Trust Report
# ════════════════════════════════════════════════════════════════════════

def api_get_trust_report(facility_id: str) -> Dict:
    """
    TRUST REPORT ENDPOINT
    Returns full trust breakdown for a specific facility.
    
    Returns:
        {
            "facility_id": ...,
            "name": ...,
            "trust_score": ...,
            "trust_grade": "A/B/C/D/F",
            "trust_flags": [...],
            "confidence": {...},
            "correction_iterations": ...,
            "extraction_summary": {...}
        }
    """
    rows = spark.table(GOLD_TABLE) \
        .filter(F.col("facility_id") == facility_id) \
        .collect()
    
    if not rows:
        return {"error": f"Facility {facility_id} not found"}
    
    row = rows[0]
    trust_score = float(row["trust_score"])
    
    # Grade assignment
    grade = "A" if trust_score >= 0.85 else \
            "B" if trust_score >= 0.70 else \
            "C" if trust_score >= 0.55 else \
            "D" if trust_score >= 0.40 else "F"
    
    extraction = json.loads(row["extraction_json"])
    trust_flags = json.loads(row["trust_flags_json"])
    confidence = json.loads(row["confidence_json"])
    
    return {
        "facility_id": facility_id,
        "name": row["name"],
        "location": f"{row['address_city']}, {row['address_stateOrRegion']}",
        "trust_score": trust_score,
        "trust_grade": grade,
        "trust_flags": trust_flags,
        "confidence": confidence,
        "correction_iterations": int(row["correction_iterations"] or 0),
        "extraction_summary": {
            "icu": extraction.get("icu", {}),
            "surgery": extraction.get("surgery", {}),
            "staff": extraction.get("staff", {}),
            "emergency": extraction.get("emergency", {}),
            "dialysis": extraction.get("dialysis", {}),
        },
        "extraction_notes": extraction.get("extraction_notes"),
        "api_version": "1.0"
    }


# ════════════════════════════════════════════════════════════════════════
# API FUNCTION 3: Desert Map
# ════════════════════════════════════════════════════════════════════════

def api_get_desert_map(
    state: Optional[str] = None,
    high_risk_only: bool = False,
    desert_type: Optional[str] = None,  # "ICU_DESERT", "DIALYSIS_DESERT", etc.
    limit: int = 100
) -> Dict:
    """
    DESERT MAP ENDPOINT
    Returns geo-tagged medical desert data.
    
    Returns:
        {
            "regions": [
                {
                    "pin_code": ...,
                    "state": ...,
                    "desert_risk_score": ...,
                    "is_high_risk": ...,
                    "desert_categories": [...],
                    "centroid_lat": ...,
                    "centroid_lon": ...,
                    "coverage": {...}
                }
            ],
            "summary": {...}
        }
    """
    df = spark.table(DESERT_TABLE)
    
    if state:
        df = df.filter(F.col("address_stateOrRegion") == state)
    if high_risk_only:
        df = df.filter(F.col("is_high_risk") == True)
    if desert_type:
        df = df.filter(F.array_contains(F.col("desert_categories"), desert_type))
    
    rows = df.orderBy(F.desc("desert_risk_score")).limit(limit).collect()
    
    regions = []
    for row in rows:
        regions.append({
            "pin_code": row["address_zipOrPostcode"],
            "state": row["address_stateOrRegion"],
            "facility_count": int(row["facility_count"]),
            "desert_risk_score": float(row["desert_risk_score"]),
            "is_high_risk": bool(row["is_high_risk"]),
            "desert_categories": list(row["desert_categories"]) if row["desert_categories"] else [],
            "centroid_lat": float(row["centroid_lat"]) if row["centroid_lat"] else None,
            "centroid_lon": float(row["centroid_lon"]) if row["centroid_lon"] else None,
            "coverage": {
                "icu": float(row["icu_coverage"]),
                "dialysis": float(row["dialysis_coverage"]),
                "emergency": float(row["emergency_coverage"]),
                "surgery": float(row["surgery_coverage"])
            },
            "avg_trust_score": float(row["avg_trust_score"])
        })
    
    # Summary statistics
    total = len(regions)
    high_risk_count = sum(1 for r in regions if r["is_high_risk"])
    
    return {
        "regions": regions,
        "summary": {
            "total_regions": total,
            "high_risk_regions": high_risk_count,
            "high_risk_percentage": round(high_risk_count / total * 100, 1) if total > 0 else 0,
            "filter_state": state,
            "filter_desert_type": desert_type
        },
        "api_version": "1.0"
    }


# ════════════════════════════════════════════════════════════════════════
# API FUNCTION 4: Full Facility Profile
# ════════════════════════════════════════════════════════════════════════

def api_get_facility_profile(facility_id: str) -> Dict:
    """
    FACILITY PROFILE ENDPOINT
    Full enriched profile including all extracted fields and source citations.
    """
    rows = spark.table(GOLD_TABLE) \
        .filter(F.col("facility_id") == facility_id) \
        .collect()
    
    if not rows:
        return {"error": f"Facility {facility_id} not found"}
    
    row = rows[0]
    extraction = json.loads(row["extraction_json"])
    
    return {
        "facility_id": facility_id,
        "name": row["name"],
        "address": {
            "city": row["address_city"],
            "state": row["address_stateOrRegion"],
            "pin_code": row["address_zipOrPostcode"]
        },
        "coordinates": {
            "latitude": float(row["latitude"]) if row["latitude"] else None,
            "longitude": float(row["longitude"]) if row["longitude"] else None
        },
        "facility_type": row["facilityTypeId"],
        "operator_type": row["operatorTypeId"],
        "trust_score": float(row["trust_score"]),
        "confidence": json.loads(row["confidence_json"]),
        "trust_flags": json.loads(row["trust_flags_json"]),
        "capabilities": extraction,
        "correction_iterations": int(row["correction_iterations"] or 0),
        "api_version": "1.0"
    }


# CELL 2 — Test all API functions
print("=== API TESTS ===\n")

# Test 1: Query
result1 = api_query_facilities(
    query="dialysis center with nephrology specialist",
    state="Uttar Pradesh",
    min_trust_score=0.5,
    top_k=3,
    include_chain_of_thought=False
)
print("Query API:")
print(f"  Recommendation: {result1.get('recommendation_summary', 'N/A')}")
print(f"  Top result: {result1.get('ranked_results', [{}])[0].get('facility_name', 'None')}\n")

# Test 2: Desert map
result2 = api_get_desert_map(state="Bihar", high_risk_only=True, limit=5)
print("Desert Map API:")
print(f"  High-risk regions in Bihar: {result2['summary']['high_risk_regions']}")
for r in result2["regions"][:3]:
    print(f"  PIN {r['pin_code']}: risk={r['desert_risk_score']:.2f}, cats={r['desert_categories']}")
```

---

## 11. MLflow 3 Tracing Integration

```python
# CELL — MLflow 3 full tracing setup
# This is the cross-cutting observability layer

import mlflow
from mlflow.entities import SpanType
import mlflow.tracking

# MLflow 3 specific: enable automatic tracing for supported frameworks
mlflow.set_experiment("/Users/<your-email>/healthcare_agent_traces")

# ── Trace decorator for all agent functions ───────────────────────────────────
def traced_agent_call(agent_name: str):
    """Decorator that wraps any agent function with MLflow tracing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with mlflow.start_run(run_name=f"{agent_name}_{int(time.time())}"):
                with mlflow.start_span(
                    name=agent_name,
                    span_type=SpanType.AGENT,
                    attributes={"agent": agent_name}
                ) as span:
                    start = time.time()
                    result = func(*args, **kwargs)
                    elapsed = (time.time() - start) * 1000
                    span.set_attribute("latency_ms", elapsed)
                    return result
        return wrapper
    return decorator


# ── Log a full extraction trace (example) ────────────────────────────────────
def log_extraction_trace(
    facility_id: str,
    input_text: str,
    extraction: Dict,
    trust_score: float,
    trust_flags: List[Dict],
    correction_iterations: int,
    tokens_used: int = 0,
    latency_ms: float = 0.0
):
    """
    Log a complete extraction event to MLflow.
    Each extraction gets its own trace for full traceability.
    """
    with mlflow.start_run(run_name=f"extraction_{facility_id[:12]}"):
        # Log inputs
        mlflow.log_param("facility_id", facility_id)
        mlflow.log_param("input_length", len(input_text))
        mlflow.log_param("llm_model", LLM_ENDPOINT)
        mlflow.log_param("correction_iterations", correction_iterations)
        
        # Log metrics
        mlflow.log_metric("trust_score", trust_score)
        mlflow.log_metric("tokens_used", tokens_used)
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("flags_count", len(trust_flags))
        mlflow.log_metric("high_severity_flags",
                          sum(1 for f in trust_flags if f["severity"] in ("HIGH", "CRITICAL")))
        
        # Log full extraction as artifact
        mlflow.log_text(json.dumps(extraction, indent=2), "extraction_output.json")
        mlflow.log_text(json.dumps(trust_flags, indent=2), "trust_flags.json")
        mlflow.log_text(input_text[:500], "input_text_truncated.txt")


# ── MLflow 3: Enable LLM Tracing ─────────────────────────────────────────────
# In MLflow 3, set the tracing exporter for observability dashboard
# This automatically captures all spans created within experiments

# View traces in Databricks UI:
# Experiments > <experiment name> > Traces tab
# Each trace shows: prompt → extraction → validation → correction chain

print("✅ MLflow 3 tracing configured")
print(f"   Experiment: /Users/<your-email>/healthcare_agent_traces")
print("   View traces: Databricks UI → Experiments → Traces tab")
```

---

## 12. LLM Prompts Reference

This section consolidates all prompts for easy tuning.

### 12.1 Extraction Prompt

Already defined in Notebook 02, CELL 3. Key design decisions:
- Temperature `0.0` for deterministic structured output
- Explicit status enum values in prompt to prevent hallucination
- `source_text` required in every sub-schema to enforce citations
- JSON-only output instruction to prevent preamble

### 12.2 Validator Prompt

Already defined in Notebook 03, CELL 2. Focuses on:
- Medical logic contradictions (not just field-level checks)
- `recommend_reextraction` boolean to control loop

### 12.3 Reasoning Prompt

Already defined in Notebook 06, CELL 1. Key features:
- Clinical framing (not generic assistant)
- `trust_score >= 0.5` hard threshold instruction
- Required JSON output with `citations[]` field
- `warnings[]` field for uncertainty surfacing

### 12.4 Corrector Prompt

Defined in Notebook 04, CELL 1. Strategy:
- Conservative correction (prefer "uncertain" over guessing)
- Provided both validator report AND original text
- Same output schema as original extractor

---

## 13. Confidence Scoring Model

The `compute_confidence()` function (Notebook 03, CELL 3) uses three components:

```
Overall Confidence = 0.40 × Completeness + 0.35 × Consistency + 0.25 × Reliability

Completeness = count(non-uncertain key fields) / 6
Consistency  = 1.0 - (0.25 × HIGH_flags) - (0.10 × MED_flags)
Reliability  = f(composite_text_length)
               └── >1000 chars → 0.85
               └── >400 chars  → 0.65
               └── >100 chars  → 0.45
               └── <100 chars  → 0.20

Confidence Interval:
  noise = 0.05 + (total_flag_count × 0.02)
  CI_low  = max(0, overall - noise)
  CI_high = min(1, overall + noise)
```

**Interpretation:**

| Overall Score | Meaning |
|---|---|
| 0.80–1.00 | High confidence — use for clinical decisions |
| 0.60–0.79 | Moderate — verify with secondary source |
| 0.40–0.59 | Low — treat as indicative only |
| 0.00–0.39 | Very low — data likely unreliable |

---

## 14. Crisis Mapping Data Pipeline

### 14.1 Geo-Ready Output Schema

The `api_get_desert_map()` function returns data structured for direct consumption by:
- **Folium** (Python map library)
- **Kepler.gl** (Databricks native)
- **Leaflet.js** (frontend)
- **Google Maps API**

### 14.2 Kepler.gl Visualization in Databricks

```python
# In a Databricks notebook, use Kepler.gl directly
from keplergl import KeplerGl
import pandas as pd

desert_pdf = spark.table(DESERT_TABLE) \
    .where(F.col("centroid_lat").isNotNull()) \
    .toPandas()

# Configure map
map_config = {
    "version": "v1",
    "config": {
        "mapStyle": {"styleType": "dark"},
        "visState": {
            "layers": [{
                "type": "heatmap",
                "config": {
                    "dataId": "deserts",
                    "columns": {"lat": "centroid_lat", "lng": "centroid_lon"},
                    "visConfig": {
                        "weightField": "desert_risk_score",
                        "colorRange": {
                            "colors": ["#00FF00", "#FFFF00", "#FF0000"]
                        }
                    }
                }
            }]
        }
    }
}

map_1 = KeplerGl(height=600, config=map_config)
map_1.add_data(data=desert_pdf[["centroid_lat", "centroid_lon", "desert_risk_score", 
                                  "is_high_risk", "address_zipOrPostcode", 
                                  "address_stateOrRegion"]], 
               name="deserts")
map_1
```

---

## Appendix: Deployment Checklist

```
□ Notebook 01: Bronze table created, row count verified
□ Notebook 01: Composite text quality > 60% records with description
□ Notebook 02: Extraction success rate > 85%
□ Notebook 02: Failed IDs logged to MLflow
□ Notebook 03: Gold table written with trust scores
□ Notebook 03: OPTIMIZE + ZORDER run on Gold table
□ Notebook 04: Self-correction run on trust_score < 0.65 records
□ Notebook 05: Vector Search endpoint ONLINE
□ Notebook 05: Index sync complete, row count matches Gold table
□ Notebook 06: Reasoning engine tested with sample queries
□ Notebook 07: Desert table written, high-risk regions identified
□ Notebook 07: Geo-JSON export written to DBFS
□ Notebook 08: All 4 API functions tested
□ MLflow: Experiments visible in Databricks UI → Traces tab
□ Delta: Change Data Feed enabled on Gold table
```

---

## Appendix: Troubleshooting

| Issue | Solution |
|---|---|
| Rate limit 429 on LLM API | Reduce `max_workers` to 2, increase sleep between batches |
| Vector Search index stuck "syncing" | Check CDF enabled: `ALTER TABLE ... SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')` |
| JSON parse errors from LLM | Increase `max_tokens`, add JSON fence stripping (already in code) |
| Low extraction quality (<70%) | Increase composite text by joining more columns; upgrade to DBRX |
| Memory errors on `collect()` | Use `limit()` + pagination; process in Spark UDFs instead |
| Gold table MERGE fails | Ensure facility_id is unique: add `.dropDuplicates(["facility_id"])` before write |

---

*Guide version 1.0 · Built for Hack-Nation × World Bank Youth Summit · Global AI Hackathon 2026*
*Databricks Free Edition · MLflow 3 · Mosaic AI Vector Search · Delta Lake*
