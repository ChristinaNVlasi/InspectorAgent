"""
SMA Logic Creator — Python Logic Engine
========================================

ARCHITECTURE
────────────
The Logic Engine is the authoritative source of all flow-construction
intelligence.  The HTML UI is responsible only for rendering and user
interaction; all judgment about what nodes exist and how they connect
is computed here.

INNER PROCESSES
────────────────
1. Intake        Receives a WizardState (industry, sections, subpaths,
                 targets, end_routing) via POST /api/flow/build.

2. Validation    Checks state completeness and cross-field consistency
                 (e.g. a sub-path cannot be active if its parent section
                 is disabled; numeric targets must be parseable).

3. Construction  The domain-specific FlowBuilder (ManufacturingFlowBuilder
                 or RemanufacturingFlowBuilder) iterates through active
                 sections and sub-paths, conditionally inserting nodes and
                 wiring edges to form a directed, connected graph.

4. Enrichment    Every agent node is matched against the AgentCatalogue.
                 The catalogue supplies: role, goal, step-by-step
                 instructions, sub-agent definitions, and a tool list.
                 Tool IDs are then resolved through the ToolCatalogue to
                 attach type and description metadata.

5. Assembly      The enriched node/edge graph is packaged into a structured
                 SMALogicBlock with meta, targets, active configuration,
                 flow steps (y-sorted), decision gates, agent manifests,
                 tool manifest, and the full edge list.

6. Serialisation Returns JSON (default) or YAML (agent-ready) depending on
                 the Accept header or ?format=yaml query parameter.

JUDGMENT METHODOLOGY
─────────────────────
Industry gate     The industry field activates the corresponding domain
                  template.  Manufacturing and Remanufacturing share the
                  same node/edge vocabulary but differ in spine structure,
                  phase names, column layout, and routing rules.

Section gate      Each top-level phase is an independent boolean toggle.
                  Disabling a section removes all its nodes and the builder
                  re-wires the spine to the nearest remaining active node.

Sub-path gate     Within each active section, individual capability nodes
                  (Scheduling, RCA, Machine Integration …) can be toggled
                  off.  The builder traces back to the last active ancestor
                  to keep the graph connected.

Decision nodes    GATE nodes are inserted automatically when their
                  triggering parent sub-path is active.  YES/NO branches
                  are wired to the next structurally correct active node.

Target values     Numeric targets (OEE %, MTTR, CO2 …) are metadata —
                  they never gate any node.  They are attached to the meta
                  block and injected as runtime context into each relevant
                  agent's instructions so the agent knows what threshold to
                  apply at runtime.

End routing       Terminal nodes (scrap, rework, quarantine, recycle …) are
                  appended outside the main spine as leaf nodes, connected
                  from the nearest quality-exit point.  End routing choices
                  do not affect the structure of the main spine.
"""

from __future__ import annotations

import datetime
import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import uvicorn
import yaml
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

# ── Node-type style palette (mirrors NT in the HTML canvas renderer) ──────────
NT: dict[str, dict[str, str]] = {
    "process":   {"color": "#4f7fff", "bg": "#0d1525", "tag": "PROC"},
    "agent":     {"color": "#a78bfa", "bg": "#120f22", "tag": "AGNT"},
    "decision":  {"color": "#f59e0b", "bg": "#1a1200", "tag": "GATE"},
    "output":    {"color": "#3ecf8e", "bg": "#0a1a12", "tag": "OUT"},
    "knowledge": {"color": "#f87171", "bg": "#1a0e0e", "tag": "KB"},
    "intake":    {"color": "#4f7fff", "bg": "#0d1525", "tag": "IN"},
    "quality":   {"color": "#2dd4bf", "bg": "#091a18", "tag": "QA"},
    "db":        {"color": "#555a66", "bg": "#111214", "tag": "DB"},
}


# ═════════════════════════════════════════════════════════════════════════════
# DOMAIN KNOWLEDGE — SECTIONS, TARGETS, END-ROUTING
# ═════════════════════════════════════════════════════════════════════════════

SECTIONS: dict[str, list[dict]] = {
    "manufacturing": [
        {
            "id": "planning", "label": "Planning", "color": "#4f7fff",
            "desc": "Work orders, prioritisation, scheduling, target input (COST · TIME · PRODUCTIVITY)",
            "subpaths": [
                {"id": "scheduling",  "label": "Scheduling agent",       "desc": "Sequence · Takt · OEE · Priority — pulls from Orders DB", "def": True},
                {"id": "material",    "label": "Material intake",        "desc": "Lots · COA · Stock check → Procurement agent on fail",   "def": True},
                {"id": "parameters",  "label": "Parameter optimisation", "desc": "AI-set Speed · Temp · Torque · pH from historical data",  "def": True},
                {"id": "qpredict",    "label": "Quality predictor",      "desc": "Monte Carlo forecast before first part runs",             "def": True},
            ],
        },
        {
            "id": "failures", "label": "Failure Analysis", "color": "#f87171",
            "desc": "Anomaly detection, RCA, machine integration, auto-fix or manual dispatch",
            "subpaths": [
                {"id": "analyse",    "label": "Analyse anomaly",     "desc": "Classify defect type · Severity gate · DB lookup",             "def": True},
                {"id": "rca",        "label": "RCA agent",           "desc": "Rank root causes by probability × risk · Top 3 surfaced",      "def": True},
                {"id": "machineint", "label": "Machine integration", "desc": "Direct PLC/SCADA command execution and YES/NO feedback",        "def": True},
                {"id": "autofix",    "label": "Auto-fix / dispatch", "desc": "Digital PLC command or AR checklist to technician",            "def": True},
            ],
        },
        {
            "id": "quality", "label": "Quality Control", "color": "#2dd4bf",
            "desc": "Real-time monitoring, deviation detection, QA inspection, release",
            "subpaths": [
                {"id": "monitoring", "label": "Monitoring agent",  "desc": "Sensors · Vision · OEE live stream against tolerance",  "def": True},
                {"id": "deviation",  "label": "Deviation triage",  "desc": "Gate → adaptive correction loop or continue",          "def": True},
                {"id": "inspection", "label": "Quality inspector", "desc": "Score · Inspect · Report per unit or batch",            "def": True},
                {"id": "release",    "label": "Release",           "desc": "Sign-off · Serial · Dispatch on QA pass",              "def": True},
            ],
        },
    ],
    "remanufacturing": [
        {
            "id": "intake", "label": "Core intake", "color": "#4f7fff",
            "desc": "Registration, traceability, disassembly, component & core identification",
            "subpaths": [
                {"id": "registration", "label": "Core registration",       "desc": "QR · Serial · Core ID validated against DB",               "def": True},
                {"id": "disassembly",  "label": "Disassembly",             "desc": "Structured disassembly guided by DB/KB",                   "def": True},
                {"id": "coreident",    "label": "Core identification",     "desc": "Core condition check against Indicators DB",               "def": True},
                {"id": "compident",    "label": "Component identification","desc": "Component-level condition check with DB/KB",               "def": True},
            ],
        },
        {
            "id": "assessment", "label": "Condition assessment", "color": "#2dd4bf",
            "desc": "Multi-indicator condition assessment using Indicators DB (COST · TIME · CO2)",
            "subpaths": [
                {"id": "condassess",  "label": "Condition assessment",      "desc": "COST · TIME · CO2 indicators from Indicators DB",          "def": True},
                {"id": "remanassess", "label": "Remanufacture assessment",  "desc": "Structural + performance score → remanufacturable?",       "def": True},
            ],
        },
        {
            "id": "decision", "label": "Strategy & execution", "color": "#f59e0b",
            "desc": "R-strategy selection, component availability, repair execution, procurement",
            "subpaths": [
                {"id": "rstrategy",  "label": "R-strategy selection",    "desc": "Repair · Reuse · Remanufacture — via ERP/DB/KB",           "def": True},
                {"id": "components", "label": "Component availability",  "desc": "YES → Allocate Parts · NO → Trigger Procurement",          "def": True},
                {"id": "execution",  "label": "Release for repair",      "desc": "Work order issued · AR guidance · Checklists",             "def": True},
            ],
        },
        {
            "id": "quality", "label": "Quality, credit & certification", "color": "#3ecf8e",
            "desc": "Quality & compliance assessment, pass/fail gate, credit release, certification report",
            "subpaths": [
                {"id": "qaassess",   "label": "Quality & compliance assessment", "desc": "Full inspection against standards",        "def": True},
                {"id": "qapass",     "label": "Quality pass gate",               "desc": "Pass → Release · Fail → Re-enter assessment", "def": True},
                {"id": "credit",     "label": "Credit release",                  "desc": "Deposit value calculated and released",    "def": True},
                {"id": "certreport", "label": "Certification report",            "desc": "Final compliance report issued",           "def": True},
            ],
        },
        {
            "id": "learning", "label": "Knowledge & feedback", "color": "#a78bfa",
            "desc": "Knowledge base update, training/feedback loop back to Industry Selector",
            "subpaths": [
                {"id": "kb",       "label": "Knowledge base update",  "desc": "Metadata · Model update · Throughput metrics",  "def": True},
                {"id": "feedback", "label": "Training/feedback loop", "desc": "Feeds back into next Remanufacturing intake cycle", "def": True},
            ],
        },
    ],
}

TARGETS: dict[str, list[dict]] = {
    "manufacturing": [
        {"id": "oee",       "label": "OEE target",         "unit": "%",    "placeholder": "e.g. 85"},
        {"id": "yield",     "label": "Yield target",        "unit": "%",    "placeholder": "e.g. 98"},
        {"id": "cycleTime", "label": "Cycle time",          "unit": "sec",  "placeholder": "e.g. 45"},
        {"id": "scrapRate", "label": "Scrap rate max",      "unit": "%",    "placeholder": "e.g. 2"},
        {"id": "mttr",      "label": "MTTR target",         "unit": "min",  "placeholder": "e.g. 30"},
        {"id": "costUnit",  "label": "Cost per unit",       "unit": "€",    "placeholder": "e.g. 12.50"},
        {"id": "mtbf",      "label": "MTBF target",         "unit": "hrs",  "placeholder": "e.g. 720"},
        {"id": "fpy",       "label": "First pass yield",    "unit": "%",    "placeholder": "e.g. 96"},
    ],
    "remanufacturing": [
        {"id": "recovery",   "label": "Recovery rate",            "unit": "%",    "placeholder": "e.g. 78"},
        {"id": "costThresh", "label": "Cost threshold",           "unit": "€",    "placeholder": "e.g. 50"},
        {"id": "condScore",  "label": "Min condition score",      "unit": "0–100","placeholder": "e.g. 45"},
        {"id": "cycleTime",  "label": "Repair cycle time",        "unit": "hrs",  "placeholder": "e.g. 4"},
        {"id": "creditVal",  "label": "Base credit value",        "unit": "€",    "placeholder": "e.g. 120"},
        {"id": "qaPasRate",  "label": "QA pass rate target",      "unit": "%",    "placeholder": "e.g. 95"},
        {"id": "co2target",  "label": "CO2 reduction target",     "unit": "kg",   "placeholder": "e.g. 12"},
        {"id": "reutilRate", "label": "Component reuse rate",     "unit": "%",    "placeholder": "e.g. 65"},
    ],
}

SCRAP_OPTIONS: dict[str, list[dict]] = {
    "manufacturing": [
        {"id": "rework",     "label": "Rework loop",          "desc": "Failed units re-enter sub-assembly station"},
        {"id": "scrap",      "label": "Scrap & log",           "desc": "Unit scrapped · Material value calculated and logged"},
        {"id": "quarantine", "label": "Quarantine hold",       "desc": "Unit isolated pending engineering review"},
        {"id": "supplier",   "label": "Supplier notification", "desc": "Bad batch flagged to supplier automatically"},
    ],
    "remanufacturing": [
        {"id": "recycle", "label": "Recycle / scrap flow",  "desc": "Material value calculated · Sent to recycler"},
        {"id": "scrap",   "label": "Scrap inventory update","desc": "Update scrap inventory · Log material yield"},
        {"id": "partial", "label": "Partial salvage",       "desc": "Recoverable components extracted before scrap"},
        {"id": "return",  "label": "Return to supplier",    "desc": "Core returned to vendor with defect report"},
    ],
}

# ═════════════════════════════════════════════════════════════════════════════
# AGENT CATALOGUE
# Every agent node in the flow is enriched with this data.
# role        — one-sentence purpose statement
# goal        — measurable success condition for the agent
# instructions— ordered list of steps the LLM agent must follow
# sub_agents  — specialist agents this agent may spawn
# tools       — tool IDs drawn from TOOL_CATALOGUE
# ═════════════════════════════════════════════════════════════════════════════

AGENT_CATALOGUE: dict[str, dict] = {
    # ── MANUFACTURING ──────────────────────────────────────────────────────
    "sched": {
        "role": "Sequences and prioritises production orders before execution begins",
        "goal": "Minimise takt-time deviation and maximise OEE across all active lines",
        "instructions": [
            "Pull open orders from ERP and rank by deadline, batch size, and changeover cost",
            "Identify bottleneck machines and pre-schedule around planned maintenance windows",
            "Re-sequence dynamically when rush orders arrive or machine availability changes",
            "Output an ordered job list with estimated start/end times and assigned resources",
        ],
        "sub_agents": [
            {"id": "takt_calculator",    "role": "Computes takt time per order from customer demand rate",          "tools": ["erp_query", "calendar_api"]},
            {"id": "bottleneck_detector","role": "Identifies capacity constraints across the line",                 "tools": ["sensor_stream_reader", "machine_status_api"]},
        ],
        "tools": ["erp_query", "machine_status_api", "calendar_api", "schedule_writer"],
    },
    "param": {
        "role": "Calculates and applies optimal machine parameters before each production run",
        "goal": "Maximise yield and minimise scrap by setting evidence-based process parameters",
        "instructions": [
            "Query historical run data for the same product/machine combination",
            "Apply statistical optimisation (Bayesian or gradient) over the parameter space",
            "Output a parameter set (speed, temperature, pressure, feed, pH, etc.) specific to this run",
            "Flag parameter sets with low historical confidence for human review before commit",
        ],
        "sub_agents": [
            {"id": "historical_retriever","role": "Retrieves past run records matching current product and machine state","tools": ["knowledge_base_query", "timeseries_db"]},
            {"id": "param_optimiser",     "role": "Runs optimisation solver over the parameter space",                   "tools": ["optimisation_solver", "simulation_runner"]},
        ],
        "tools": ["knowledge_base_query", "timeseries_db", "optimisation_solver", "plc_write", "scada_write"],
    },
    "qpred": {
        "role": "Predicts output quality before the first part runs using Monte Carlo simulation",
        "goal": "Surface high-risk production orders for human review before defects occur",
        "instructions": [
            "Load proposed parameter set and current material/tooling state",
            "Run Monte Carlo simulation (≥1000 iterations) sampling uncertainty distributions",
            "Output predicted yield %, defect rate, and confidence interval",
            "Raise a risk flag and pause for human approval if predicted yield falls below threshold",
        ],
        "sub_agents": [
            {"id": "monte_carlo_engine","role": "Runs stochastic quality simulation",                                      "tools": ["simulation_runner", "uncertainty_model"]},
            {"id": "risk_classifier",  "role": "Classifies forecast outputs into LOW / MEDIUM / HIGH risk bands",          "tools": ["threshold_evaluator"]},
        ],
        "tools": ["simulation_runner", "uncertainty_model", "threshold_evaluator", "human_review_trigger"],
    },
    "anal": {
        "role": "Classifies detected anomalies by type and severity before routing to RCA",
        "goal": "Correctly categorise every anomaly to ensure the right resolution path is taken",
        "instructions": [
            "Receive anomaly signal with sensor ID, reading value, and timestamp",
            "Classify defect type (vibration, thermal, dimensional, vision, process) using trained classifier",
            "Assign severity level (LOW / MEDIUM / HIGH / CRITICAL) based on distance from tolerance and rate of change",
            "Route CRITICAL anomalies directly to emergency stop; route others to RCA agent",
        ],
        "sub_agents": [
            {"id": "defect_classifier","role": "Classifies anomaly type from sensor signature",                          "tools": ["ml_classifier", "feature_extractor"]},
            {"id": "severity_scorer",  "role": "Scores anomaly severity from deviation magnitude and trend",             "tools": ["trend_analyser", "threshold_evaluator"]},
        ],
        "tools": ["ml_classifier", "feature_extractor", "trend_analyser", "threshold_evaluator", "alert_publisher"],
    },
    "rca": {
        "role": "Identifies and ranks the most probable root causes of a detected anomaly",
        "goal": "Surface the top 3 root causes with probability scores so the correct fix is applied first",
        "instructions": [
            "Collect sensor snapshot, log buffers, and recent parameter changes from the anomaly window",
            "Compare current anomaly signature against the historical failure pattern library",
            "Generate candidate root cause list and score each by probability × risk impact",
            "Present top 3 causes with supporting evidence to the auto-fix gate or operator UI",
            "If no historical match exists, escalate to human expert and flag as new pattern",
        ],
        "sub_agents": [
            {"id": "pattern_matcher","role": "Matches current anomaly signature against historical failure patterns","tools": ["knowledge_base_query", "vector_similarity_search"]},
            {"id": "cause_ranker",  "role": "Ranks candidate causes by Bayesian probability × risk score",          "tools": ["probability_model", "risk_scorer"]},
        ],
        "tools": ["knowledge_base_query", "vector_similarity_search", "probability_model", "risk_scorer", "timeseries_db", "human_review_trigger"],
    },
    "machint": {
        "role": "Interfaces directly with machine controllers to execute corrective commands",
        "goal": "Apply corrective actions to PLC/SCADA systems with confirmation of execution",
        "instructions": [
            "Receive corrective command from auto-fix gate (parameter ID, target value, direction)",
            "Validate command against safe operating limits before writing",
            "Write command to PLC or SCADA and await acknowledgement within timeout window",
            "Confirm execution by reading back the updated parameter value",
            "Log command, acknowledgement, and confirmation timestamp to event store",
        ],
        "sub_agents": [],
        "tools": ["plc_write", "plc_read", "scada_write", "scada_read", "safe_limit_validator", "event_logger"],
    },
    "cmd": {
        "role": "Executes an automated correction command on the machine controller",
        "goal": "Apply the identified fix immediately without halting production",
        "instructions": [
            "Receive fix specification (parameter, target value, correction type) from RCA agent",
            "Route to machine integration layer for PLC/SCADA execution",
            "Monitor output for one cycle to confirm anomaly is cleared",
            "Report resolution status back to validation gate",
        ],
        "sub_agents": [],
        "tools": ["plc_write", "scada_write", "plc_read", "event_logger"],
    },
    "dispatch": {
        "role": "Dispatches a technician with a structured AR-guided checklist for manual fixes",
        "goal": "Ensure the right technician receives the right instructions within response SLA",
        "instructions": [
            "Receive manual fix recommendation and root cause summary from RCA agent",
            "Select appropriate AR checklist template for the identified fault type",
            "Assign to available technician based on skill match and proximity",
            "Push checklist to technician AR device with fault location, steps, and required tools",
            "Track checklist completion and feed result back to resolution validation gate",
        ],
        "sub_agents": [
            {"id": "checklist_builder",     "role": "Assembles step-by-step AR repair checklist from fault type template","tools": ["checklist_template_store", "ar_content_renderer"]},
            {"id": "technician_dispatcher", "role": "Finds and assigns the best available technician",                     "tools": ["workforce_management_api", "notification_service"]},
        ],
        "tools": ["checklist_template_store", "ar_content_renderer", "workforce_management_api", "notification_service", "task_tracker"],
    },
    "optim": {
        "role": "Tunes process parameters and retrains models after each production cycle",
        "goal": "Compound improvement cycle-over-cycle by feeding outcomes back into the intelligence layer",
        "instructions": [
            "Collect cycle outcome data: yield, scrap rate, parameter drift, correction history",
            "Run parameter auto-tuning to tighten setpoints based on observed performance",
            "Submit new labelled examples to model retraining pipeline",
            "Update KPI dashboard with latest OEE, MTTR, FPY, and cost-per-unit metrics",
            "Generate shift report summarising deviations, corrections, and improvement trends",
        ],
        "sub_agents": [
            {"id": "model_retrainer", "role": "Submits new outcome data to model fine-tuning pipeline","tools": ["ml_training_api", "dataset_writer"]},
            {"id": "kpi_aggregator",  "role": "Computes and publishes updated KPI metrics",           "tools": ["timeseries_db", "kpi_dashboard_writer"]},
        ],
        "tools": ["timeseries_db", "kpi_dashboard_writer", "ml_training_api", "dataset_writer", "report_generator"],
    },
    "proc": {
        "role": "Handles procurement of materials or components when stock is insufficient",
        "goal": "Restore stock levels without delaying the production schedule",
        "instructions": [
            "Receive stock-out alert with item ID, required quantity, and required-by date",
            "Query approved supplier list and select best option by lead time and cost",
            "Raise purchase order in ERP and notify procurement team",
            "Track order status and alert scheduler if lead time risks production delay",
        ],
        "sub_agents": [],
        "tools": ["erp_query", "erp_update", "supplier_api", "notification_service"],
    },
    # ── REMANUFACTURING ────────────────────────────────────────────────────
    "rstrat": {
        "role": "Selects the optimal circular economy strategy for each incoming core",
        "goal": "Maximise material value recovery while minimising cost-to-recovery ratio",
        "instructions": [
            "Receive condition assessment score, cost estimate, and component availability status",
            "Evaluate all viable R-strategies (Repair, Reuse, Remanufacture, Recycle) against cost-recovery threshold",
            "Query ERP/DB/KB for current demand, inventory levels, and carbon impact per strategy",
            "Select the strategy with the best value score and assign it to the work order",
            "Flag cores with unknown defect types for human technician review before proceeding",
        ],
        "sub_agents": [
            {"id": "strategy_evaluator","role": "Scores each R-strategy option against cost, time, and CO2 indicators","tools": ["indicators_db_query", "cost_model"]},
            {"id": "demand_checker",    "role": "Checks remanufactured unit demand and inventory levels",              "tools": ["erp_query", "inventory_api"]},
        ],
        "tools": ["indicators_db_query", "erp_query", "inventory_api", "cost_model", "knowledge_base_query", "work_order_writer"],
    },
    "trig_proc": {
        "role": "Triggers procurement of missing components when inventory check fails",
        "goal": "Source required parts without delaying the repair execution schedule",
        "instructions": [
            "Receive stock-out alert with part ID, required quantity, and core work order ID",
            "Query approved supplier list for the part and select by lead time and cost",
            "Raise purchase order in ERP and link it to the core work order",
            "Monitor delivery and update component availability status when stock arrives",
        ],
        "sub_agents": [],
        "tools": ["erp_query", "erp_update", "supplier_api", "inventory_api", "notification_service"],
    },
    "alloc": {
        "role": "Allocates in-stock components to the active repair work order",
        "goal": "Reserve and stage the correct parts before repair execution begins",
        "instructions": [
            "Confirm component availability in inventory system",
            "Reserve required quantities against the work order ID",
            "Generate pick list and trigger warehouse staging task",
            "Update inventory records and work order status",
        ],
        "sub_agents": [],
        "tools": ["inventory_api", "erp_update", "warehouse_management_api", "work_order_writer"],
    },
    # ── SHARED ─────────────────────────────────────────────────────────────
    "rpt_gen": {
        "role": "Generates production or remanufacturing performance reports",
        "goal": "Produce accurate, timely reports that reflect the current operational state",
        "instructions": [
            "Aggregate KPI data for the completed cycle or shift",
            "Format report according to defined template (shift summary, batch report, or KPI dashboard)",
            "Distribute report to configured recipients via notification service",
            "Archive report to document management system with timestamp and run ID",
        ],
        "sub_agents": [],
        "tools": ["kpi_dashboard_writer", "report_generator", "notification_service", "document_store"],
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# TOOL CATALOGUE
# Every tool referenced by an agent must be declared here.
# ═════════════════════════════════════════════════════════════════════════════

TOOL_CATALOGUE: dict[str, dict] = {
    "erp_query":               {"type": "data_retrieval",  "description": "Queries ERP system for orders, BOMs, inventory, and production records"},
    "erp_update":              {"type": "data_write",      "description": "Writes updates to ERP (order status, inventory levels, work orders)"},
    "machine_status_api":      {"type": "data_retrieval",  "description": "Returns real-time availability, calibration, and health status of machines"},
    "iolink_reader":           {"type": "sensor_stream",   "description": "Subscribes to IO-Link sensor streams and returns normalised readings"},
    "plc_reader":              {"type": "sensor_stream",   "description": "Reads current parameter values from PLC registers"},
    "plc_read":                {"type": "sensor_stream",   "description": "Reads current parameter values from PLC registers"},
    "plc_write":               {"type": "actuation",       "description": "Writes parameter values to PLC registers with acknowledgement"},
    "scada_reader":            {"type": "sensor_stream",   "description": "Reads process variables from SCADA system"},
    "scada_read":              {"type": "sensor_stream",   "description": "Reads process variables from SCADA system"},
    "scada_write":             {"type": "actuation",       "description": "Sends setpoint commands to SCADA-controlled equipment"},
    "camera_feed":             {"type": "sensor_stream",   "description": "Returns latest frame or frame sequence from production cameras"},
    "vision_model_inference":  {"type": "ai_model",        "description": "Runs defect detection or dimensional measurement on an image input"},
    "ml_classifier":           {"type": "ai_model",        "description": "Classifies input features into defined anomaly or defect categories"},
    "ml_training_api":         {"type": "ai_model",        "description": "Submits labelled examples to the model retraining pipeline"},
    "optimisation_solver":     {"type": "ai_model",        "description": "Runs parameter optimisation over a defined search space"},
    "simulation_runner":       {"type": "ai_model",        "description": "Executes Monte Carlo or process simulation with given inputs"},
    "probability_model":       {"type": "ai_model",        "description": "Computes posterior probability scores for candidate root causes"},
    "knowledge_base_query":    {"type": "data_retrieval",  "description": "Searches the SMA knowledge base for historical patterns, manuals, and resolutions"},
    "vector_similarity_search":{"type": "data_retrieval",  "description": "Finds nearest-neighbour matches in the failure pattern embedding store"},
    "timeseries_db":           {"type": "data_retrieval",  "description": "Queries time-series data store for sensor history and trend data"},
    "dataset_writer":          {"type": "data_write",      "description": "Writes new labelled training examples to the ML dataset store"},
    "threshold_evaluator":     {"type": "rule_engine",     "description": "Evaluates a value against defined upper/lower thresholds and returns a severity band"},
    "rule_engine":             {"type": "rule_engine",     "description": "Applies configured business rules to an input and returns a routing decision"},
    "safe_limit_validator":    {"type": "rule_engine",     "description": "Validates a proposed actuation command against machine safe operating limits"},
    "alert_publisher":         {"type": "messaging",       "description": "Publishes structured alert events to the SMA event bus"},
    "notification_service":    {"type": "messaging",       "description": "Sends notifications to operators, technicians, or systems via configured channels"},
    "human_review_trigger":    {"type": "messaging",       "description": "Pauses the flow and requests human review via the operator UI"},
    "event_logger":            {"type": "data_write",      "description": "Appends structured events (corrections, decisions, outcomes) to the audit log"},
    "report_generator":        {"type": "data_write",      "description": "Produces formatted reports (PDF, JSON, HTML) from aggregated data inputs"},
    "kpi_dashboard_writer":    {"type": "data_write",      "description": "Updates the live KPI dashboard with latest metrics"},
    "qa_report_writer":        {"type": "data_write",      "description": "Writes unit-level or batch-level QA inspection reports"},
    "serial_number_generator": {"type": "data_write",      "description": "Generates and assigns unique serial numbers or batch IDs"},
    "dispatch_trigger":        {"type": "actuation",       "description": "Triggers downstream logistics, WIP transfer, or dispatch actions"},
    "schedule_writer":         {"type": "data_write",      "description": "Writes the optimised job sequence back to the scheduling system"},
    "work_order_writer":       {"type": "data_write",      "description": "Creates or updates work orders in the production management system"},
    "checklist_template_store":{"type": "data_retrieval",  "description": "Retrieves AR repair checklist templates by fault type"},
    "ar_content_renderer":     {"type": "actuation",       "description": "Pushes structured AR content to the technician AR device"},
    "workforce_management_api":{"type": "data_retrieval",  "description": "Returns available technician list with skills and location"},
    "task_tracker":            {"type": "data_write",      "description": "Creates and tracks technician task assignments to completion"},
    "document_store":          {"type": "data_write",      "description": "Archives documents and reports with metadata and timestamp"},
    "scoring_model":           {"type": "ai_model",        "description": "Aggregates sub-inspection scores into a composite quality score"},
    "cmm_reader":              {"type": "sensor_stream",   "description": "Reads dimensional measurement data from CMM or gauge systems"},
    "feature_extractor":       {"type": "ai_model",        "description": "Extracts structured feature vectors from raw sensor or image data"},
    "trend_analyser":          {"type": "ai_model",        "description": "Computes trend direction and rate-of-change from time-series input"},
    "risk_scorer":             {"type": "ai_model",        "description": "Assigns a risk impact score to a candidate root cause or anomaly"},
    "cost_model":              {"type": "ai_model",        "description": "Estimates repair cost, labour time, and margin for a given core condition"},
    "indicators_db_query":     {"type": "data_retrieval",  "description": "Queries the Indicators DB for COST · TIME · CO2 benchmarks per strategy"},
    "inventory_api":           {"type": "data_retrieval",  "description": "Returns real-time stock levels and reservation status for parts and materials"},
    "warehouse_management_api":{"type": "actuation",       "description": "Triggers pick, stage, and transfer tasks in the warehouse management system"},
    "supplier_api":            {"type": "data_retrieval",  "description": "Queries approved supplier catalogue for lead times, pricing, and availability"},
    "calendar_api":            {"type": "data_retrieval",  "description": "Returns maintenance windows, shift schedules, and production calendar data"},
    "uncertainty_model":       {"type": "ai_model",        "description": "Returns uncertainty distributions for process parameters from historical data"},
    "sensor_stream_reader":    {"type": "sensor_stream",   "description": "Generic sensor stream reader for IO-Link and SCADA integrations"},
}


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    id: str
    type: str
    label: str
    desc: str
    x: float
    y: float
    color: str = ""
    bg: str = ""
    tag: str = ""

    def __post_init__(self) -> None:
        if not self.color and self.type in NT:
            self.color = NT[self.type]["color"]
            self.bg    = NT[self.type]["bg"]
            self.tag   = NT[self.type]["tag"]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Edge:
    from_id: str
    to_id:   str
    type:    str = "main"  # main | yes | no | db | feedback

    def to_dict(self) -> dict:
        return {"from": self.from_id, "to": self.to_id, "type": self.type}


# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS (API I/O)
# ═════════════════════════════════════════════════════════════════════════════

class WizardState(BaseModel):
    """
    Represents the complete state collected by the 5-step wizard in the UI.

    Fields
    ------
    industry    : "manufacturing" or "remanufacturing"
    sections    : {section_id: bool}  — which major phases are active
    subpaths    : {section_id: {subpath_id: bool}}  — granularity within phases
    targets     : {target_id: str}  — operator-supplied numeric KPI values
    end_routing : {route_id: bool}  — which terminal paths are active
    """
    industry:    str
    sections:    dict[str, bool]             = {}
    subpaths:    dict[str, dict[str, bool]]  = {}
    targets:     dict[str, str]              = {}
    end_routing: dict[str, bool]             = {}


class FlowResponse(BaseModel):
    """
    Full response returned to the UI after flow construction.

    nodes       : list of node objects ready for canvas rendering
    edges       : list of edge objects ready for SVG rendering
    logic_block : fully assembled SMALogicBlock as a dict
    """
    nodes:       list[dict]
    edges:       list[dict]
    logic_block: dict


# ═════════════════════════════════════════════════════════════════════════════
# BASE FLOW BUILDER
# ═════════════════════════════════════════════════════════════════════════════

class FlowBuilder:
    """
    Base class for domain-specific flow builders.

    Judgment helpers
    ----------------
    _sp(section, path)  Returns True unless the wizard explicitly set the
                        sub-path to False.  Defaults to True when the key is
                        absent (matching wizard default behaviour).

    _sec(section)       Returns True unless the wizard explicitly disabled the
                        section.

    _find(node_id)      Returns the Node object with the given id, or None.
    """

    def __init__(self, state: WizardState) -> None:
        self.state  = state
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    # ── Sub-path / section active checks ─────────────────────────────────────
    def _sp(self, section: str, path: str) -> bool:
        """Return True if sub-path is active (default True when unset)."""
        return self.state.subpaths.get(section, {}).get(path, True) is not False

    def _sec(self, section: str) -> bool:
        """Return True if section is active (default True when unset)."""
        return self.state.sections.get(section, True) is not False

    # ── Node / edge helpers ───────────────────────────────────────────────────
    def _add(self, node_id: str, ntype: str, label: str, desc: str,
             x: float, y: float) -> Node:
        n = Node(id=node_id, type=ntype, label=label, desc=desc, x=x, y=y)
        self.nodes.append(n)
        return n

    def _edge(self, from_id: str, to_id: str, etype: str = "main") -> None:
        self.edges.append(Edge(from_id=from_id, to_id=to_id, type=etype))

    def _find(self, node_id: str) -> Optional[Node]:
        return next((n for n in self.nodes if n.id == node_id), None)

    def build(self) -> tuple[list[Node], list[Edge]]:
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════════════════
# MANUFACTURING FLOW BUILDER
# Layout:
#   Col A (x=100)  — Planning
#   Col B (x=480)  — Failure Analysis
#   Col C (x=820)  — Quality Control
#   Shared bottom  — Target Achieved → KB → Feedback
# ═════════════════════════════════════════════════════════════════════════════

class ManufacturingFlowBuilder(FlowBuilder):
    """
    Builds the three-column manufacturing flow graph.

    Judgment rules applied here
    ───────────────────────────
    1. All three section headers (Planning / Failure Analysis / Quality Control)
       are always rendered as structural anchors even if their sub-paths are all
       disabled — the columns must remain identifiable.

    2. Within Planning, each sub-path is conditional.  The 'material' gate is a
       decision node; its NO branch always spawns a Procurement Agent side-node.

    3. Within Failure Analysis, the YES/NO fork at Machine Integration drives
       two parallel branches (auto-correction vs AR dispatch).  Both converge at
       Optimisation.

    4. Quality Control drives the QA gate.  A NO exit from the QA gate feeds
       back into the RCA column (if active) via a Report Generation node.

    5. The shared bottom block (Target Achieved gate → Report / Parameter Tuning
       → KB → feedback loop) is always present regardless of section toggles.
    """

    COL_A  = 100   # Planning
    COL_B  = 480   # Failure Analysis
    COL_C  = 820   # Quality Control
    ROW_H  = 120

    def build(self) -> tuple[list[Node], list[Edge]]:
        self._build_spine()
        self._build_planning()
        self._build_failure_analysis()
        self._build_quality_control()
        self._build_shared_bottom()
        self._build_end_routing()
        return self.nodes, self.edges

    # ── Spine (industry selector + header) ───────────────────────────────────
    def _build_spine(self) -> None:
        cx = self.COL_B + 50
        self._add("industry", "process", "Industry Selector",  "Manufacturing sector selected",  cx, 20)
        self._add("mfg",      "process", "Manufacturing",      "Active manufacturing sector",    cx, 140)
        self._edge("industry", "mfg")

        # Column section headers
        self._add("hdr-plan", "quality",   "Planning",         "Prioritisation · Scheduling · Target input", self.COL_A, 270)
        self._add("hdr-fail", "knowledge", "Failure Analysis", "Analyse · RCA · Machine Integration",        self.COL_B, 270)
        self._add("hdr-qual", "quality",   "Quality Control",  "Inspection · Quality pass · Release",        self.COL_C, 270)
        self._edge("mfg", "hdr-plan")
        self._edge("mfg", "hdr-fail")
        self._edge("mfg", "hdr-qual")

        # Sidebar DB nodes (always present)
        self._add("orders-db",    "db", "Orders",                              "ERP production orders",         self.COL_A - 160, 290)
        self._add("target-input", "db", "Target Input\nCOST · TIME\nPRODUCTIVITY", "Operator KPI targets",    self.COL_A - 160, 400)

    # ── Planning column ───────────────────────────────────────────────────────
    def _build_planning(self) -> None:
        y = 270 + self.ROW_H  # start just below header

        # Prioritisation is always present (it's the planning entry point)
        self._add("prioritise", "agent", "Prioritisation",
                  "Orders ranked by COST · TIME · PRODUCTIVITY",
                  self.COL_A, y)
        self._edge("hdr-plan",    "prioritise")
        self._edge("orders-db",   "prioritise", "db")
        self._edge("target-input","prioritise", "db")
        y += self.ROW_H
        prev = "prioritise"

        # Scheduling
        if self._sp("planning", "scheduling"):
            self._add("sched", "agent", "Scheduling",
                      "Sequence · Takt · OEE · Priority",
                      self.COL_A, y)
            self._edge(prev, "sched")
            y += self.ROW_H
            prev = "sched"

        # Material gate
        if self._sp("planning", "material"):
            self._add("mat-gate", "decision", "Material / Resource OK?",
                      "Stock · Tooling · Calibration check",
                      self.COL_A, y)
            self._edge(prev, "mat-gate")
            self._add("proc", "agent", "Procurement Agent",
                      "Reorder · Substitute · Hold",
                      self.COL_A - 200, y + 20)
            self._edge("mat-gate", "proc", "no")
            y += self.ROW_H
            prev = "mat-gate"
            edge_type = "yes"
        else:
            edge_type = "main"

        # Parameter optimisation
        if self._sp("planning", "parameters"):
            self._add("param", "agent", "Parameter Optimisation",
                      "Speed · Temp · Torque · pH — AI-set",
                      self.COL_A, y)
            self._edge(prev, "param", edge_type)
            y += self.ROW_H
            prev = "param"
            edge_type = "main"

        # Quality predictor
        if self._sp("planning", "qpredict"):
            self._add("qpred", "agent", "Quality Predictor",
                      "Monte Carlo · Risk flag · High-risk review",
                      self.COL_A, y)
            self._edge(prev, "qpred", edge_type)
            y += self.ROW_H

        self._p_bottom = y  # expose for shared bottom

    # ── Failure Analysis column ───────────────────────────────────────────────
    def _build_failure_analysis(self) -> None:
        y = 270 + self.ROW_H
        prev = "hdr-fail"

        # Side DB for RCA
        self._add("rca-db", "db", "DB", "Historical failure patterns", self.COL_B - 160, y + 20)

        if self._sp("failures", "analyse"):
            self._add("anal", "agent", "Analyse",
                      "Classify defect type · Severity · DB lookup",
                      self.COL_B, y)
            self._edge(prev, "anal")
            y += self.ROW_H
            prev = "anal"

        if self._sp("failures", "rca"):
            self._add("rca", "agent", "RCA",
                      "Rank root causes · Top 3 · Evidence links",
                      self.COL_B, y)
            self._edge(prev, "rca")
            self._edge("rca-db", "rca", "db")
            y += self.ROW_H
            prev = "rca"

        if self._sp("failures", "machineint"):
            self._add("machint", "decision", "Machine Integration",
                      "YES → auto command · NO → dispatch",
                      self.COL_B, y)
            self._edge(prev, "machint")
            y += self.ROW_H
            prev = "machint"

        if self._sp("failures", "autofix"):
            self._add("cmd", "agent", "Auto-correction",
                      "PLC · SCADA · Adjust setpoint",
                      self.COL_B, y)
            if self._find("machint"):
                self._edge("machint", "cmd", "yes")
            else:
                self._edge(prev, "cmd")

            self._add("dispatch", "agent", "Dispatch + AR guide",
                      "AR checklist · Skill match · Task assign",
                      self.COL_B + 220, y - self.ROW_H + 20)
            if self._find("machint"):
                self._edge("machint", "dispatch", "no")
            y += self.ROW_H
            prev = "cmd"

        self._add("optim", "agent", "Optimisation",
                  "Tune params · KPI update · Model retrain",
                  self.COL_B, y)
        self._edge(prev, "optim")
        y += self.ROW_H

        self._f_bottom = y

    # ── Quality Control column ────────────────────────────────────────────────
    def _build_quality_control(self) -> None:
        y = 270 + self.ROW_H
        prev = "hdr-qual"

        if self._sp("quality", "inspection"):
            self._add("qinsp", "agent", "Quality Inspector",
                      "Score · Inspect · Vision · Report",
                      self.COL_C, y)
            self._edge(prev, "qinsp")
            y += self.ROW_H
            prev = "qinsp"

        self._add("qa-gate", "decision", "Quality Pass",
                  "Score threshold · Spec conformance",
                  self.COL_C, y)
        self._edge(prev, "qa-gate")
        y += self.ROW_H

        # NO branch → report → feeds back into RCA if active
        self._add("rpt-qa", "output", "Report Generation",
                  "QA fail report · Defect summary",
                  self.COL_C + 220, y - self.ROW_H + 20)
        self._edge("qa-gate", "rpt-qa", "no")
        if self._find("rca"):
            self._edge("rpt-qa", "rca")

        if self._sp("quality", "release"):
            self._add("release", "output", "Release",
                      "Sign-off · Serial · Dispatch",
                      self.COL_C, y)
            self._edge("qa-gate", "release", "yes")
            y += self.ROW_H

        self._q_bottom = y

    # ── Shared bottom block ───────────────────────────────────────────────────
    def _build_shared_bottom(self) -> None:
        shared_y = max(self._p_bottom, self._f_bottom, self._q_bottom) + self.ROW_H
        mid_x    = self.COL_B + 50

        self._add("tgt-gate", "decision", "Target Achieved",
                  "KPI vs threshold · All targets met?",
                  mid_x, shared_y)
        if self._find("release"):
            self._edge("release", "tgt-gate")
        self._edge("optim", "tgt-gate")

        # NO → Report Generation (left)
        self._add("rpt-tgt", "output", "Report Generation",
                  "KPI summary · Targets archive",
                  mid_x - 240, shared_y + 20)
        self._edge("tgt-gate", "rpt-tgt", "no")

        # YES → Parameter Tuning (right)
        self._add("param-tune", "agent", "Parameter Tuning",
                  "Fine-tune setpoints · Historical calibration",
                  mid_x + 240, shared_y + 20)
        self._edge("tgt-gate", "param-tune", "yes")

        kb_y = shared_y + self.ROW_H + 20
        self._add("kb", "knowledge",
                  "Knowledge Base\n& Training/Feedback loop",
                  "Neural cache · Model retrain · Mfg DB · Dashboard",
                  mid_x, kb_y)
        self._edge("rpt-tgt",    "kb")
        self._edge("param-tune", "kb")
        self._edge("kb", "mfg", "feedback")  # feedback loop

    # ── End routing ───────────────────────────────────────────────────────────
    def _build_end_routing(self) -> None:
        defs = {
            "rework":     {"label": "Rework loop",    "desc": "Failed units re-enter sub-assembly"},
            "scrap":      {"label": "Scrap & log",    "desc": "Unit scrapped · Material value logged"},
            "quarantine": {"label": "Quarantine",     "desc": "Hold for engineering review"},
            "supplier":   {"label": "Supplier alert", "desc": "Batch flagged to supplier automatically"},
        }
        end_x = self.COL_C + 240
        ey = self._q_bottom + self.ROW_H
        for rid, active in self.state.end_routing.items():
            if not active:
                continue
            d = defs.get(rid, {"label": rid, "desc": ""})
            self._add(f"end-{rid}", "output", d["label"], d["desc"], end_x, ey)
            ey += 90


# ═════════════════════════════════════════════════════════════════════════════
# REMANUFACTURING FLOW BUILDER
# Single vertical spine with side nodes.
# ═════════════════════════════════════════════════════════════════════════════

class RemanufacturingFlowBuilder(FlowBuilder):
    """
    Builds the single-spine remanufacturing flow graph.

    Judgment rules applied here
    ───────────────────────────
    1. The spine runs top-to-bottom: Industry → Core ID → Disassembly →
       Component ID → parallel condition checks → Condition Assessment →
       Reman Assessment → Is Remanufacturable? gate → … → KB.

    2. The 'Is Remanufacturable?' decision gate is inserted only when
       'remanassess' sub-path is active; its NO branch always feeds a
       Recycle/Scrap Flow leaf node.

    3. The Component Availability gate drives a YES/NO fork: YES allocates
       parts; NO triggers procurement.  Both paths converge at Release for
       Repair.

    4. The Quality pass gate (if active) loops back to Quality & Compliance
       Assessment on failure, with a hard escalation after a second fail.

    5. The Knowledge Base + feedback loop is always the terminal node
       regardless of section toggles.
    """

    CX    = 200   # spine x-centre
    ROW_H = 120

    def build(self) -> tuple[list[Node], list[Edge]]:
        self._y = 20  # running y-position on the spine
        self._build_spine_top()
        self._build_core_intake()
        self._build_condition_assessment()
        self._build_strategy_execution()
        self._build_quality_certification()
        self._build_knowledge_feedback()
        self._build_end_routing()
        return self.nodes, self.edges

    def _adv(self, n: int = 1) -> None:
        """Advance spine y position by n row-heights."""
        self._y += n * self.ROW_H

    # ── Spine top ─────────────────────────────────────────────────────────────
    def _build_spine_top(self) -> None:
        self._add("industry",   "process", "Industry Selector",  "Remanufacturing sector selected", self.CX, self._y); self._adv()
        self._add("reman-hdr",  "process", "Remanufacturing",    "Circular economy core recovery",  self.CX, self._y); self._adv()
        self._edge("industry", "reman-hdr")

    # ── Core intake ───────────────────────────────────────────────────────────
    def _build_core_intake(self) -> None:
        if not self._sec("intake"):
            return
        sp = "intake"

        self._add("core-db", "db", "DB", "Core validation database", self.CX - 170, self._y + 20)
        self._add("core-id", "process", "Core Identification",
                  "QR · Serial · Core ID validated vs DB", self.CX, self._y)
        self._edge("reman-hdr", "core-id")
        self._edge("core-db",   "core-id", "db")
        self._adv()
        prev = "core-id"

        if self._sp(sp, "disassembly"):
            self._add("disass", "process", "Disassembly",
                      "Structured disassembly guided by DB/KB", self.CX, self._y)
            self._edge(prev, "disass")
            self._add("dis-db", "db", "DB/KB", "Disassembly knowledge base", self.CX - 170, self._y + 20)
            self._edge("dis-db", "disass", "db")
            self._adv()
            prev = "disass"

        if self._sp(sp, "compident"):
            self._add("comp-id", "process", "Component Identification",
                      "Component-level condition check", self.CX, self._y)
            self._edge(prev, "comp-id")
            self._add("comp-db", "db", "DB/KB", "Component knowledge base", self.CX - 170, self._y + 20)
            self._edge("comp-db", "comp-id", "db")
            self._adv()
            prev = "comp-id"

        if self._sp(sp, "coreident"):
            check_y = self._y
            self._add("comp-cond", "process", "Component Condition Check",
                      "Component degradation & wear assessment", self.CX - 160, check_y)
            self._add("core-cond", "process", "Core Condition Check",
                      "Structural integrity & geometry scan",   self.CX + 160, check_y)
            self._edge(prev, "comp-cond")
            self._edge(prev, "core-cond")
            self._adv()

    # ── Condition assessment ──────────────────────────────────────────────────
    def _build_condition_assessment(self) -> None:
        if not self._sec("assessment"):
            return
        sp = "assessment"

        has_checks = (self._sec("intake") and self._sp("intake", "coreident"))

        if self._sp(sp, "condassess"):
            self._add("condass", "output", "Condition Assessment",
                      "COST · TIME · CO2 indicators evaluated", self.CX, self._y)
            if has_checks:
                self._edge("comp-cond", "condass")
                self._edge("core-cond", "condass")
            else:
                fallback = next(
                    (nid for nid in ("comp-id", "disass", "reman-hdr") if self._find(nid)),
                    "reman-hdr"
                )
                self._edge(fallback, "condass")
            self._add("ind-db", "db",
                      "Indicators DB\nCOST · TIME\nCO2",
                      "KPI indicators reference",
                      self.CX + 210, self._y + 20)
            self._edge("ind-db", "condass", "db")
            self._adv()

        if self._sp(sp, "remanassess"):
            self._add("reman-ass", "process", "Reman Assessment",
                      "Structural + performance score", self.CX, self._y)
            prev = next(
                (nid for nid in ("condass", "core-cond", "reman-hdr") if self._find(nid)),
                "reman-hdr"
            )
            self._edge(prev, "reman-ass")
            self._adv()

            self._add("reman-gate", "decision", "Is Remanufacturable?",
                      "Cost · Condition threshold check", self.CX, self._y)
            self._edge("reman-ass", "reman-gate")
            self._adv()

            self._add("recycle", "output", "Recycle / Scrap Flow",
                      "Material value calc · Sent to recycler",
                      self.CX - 180, self._y - self.ROW_H + 20)
            self._edge("reman-gate", "recycle", "no")

    # ── Strategy & execution ──────────────────────────────────────────────────
    def _build_strategy_execution(self) -> None:
        if not self._sec("decision"):
            return
        sp = "decision"
        gate = self._find("reman-gate")
        prev_d = "reman-gate" if gate else next(
            (nid for nid in ("reman-ass", "condass", "reman-hdr") if self._find(nid)),
            "reman-hdr"
        )

        if self._sp(sp, "rstrategy"):
            self._add("rstrat", "agent", "R-Strategy Selection",
                      "Repair · Reuse · Remanufacture decision", self.CX, self._y)
            self._edge(prev_d, "rstrat", "yes" if gate else "main")
            self._add("erp-db", "db", "ERP/DB/KB", "Strategy reference data", self.CX + 210, self._y + 20)
            self._edge("erp-db", "rstrat", "db")
            self._adv()
            prev_d = "rstrat"

        if self._sp(sp, "components"):
            self._add("comp-avail", "decision", "Component Availability",
                      "In-stock check against work order", self.CX, self._y)
            self._edge(prev_d, "comp-avail")
            self._adv()
            prev_d = "comp-avail"

            self._add("trig-proc", "agent", "Trigger Procurement",
                      "Reorder · Substitute · Lead-time track",
                      self.CX + 240, self._y - self.ROW_H + 20)
            self._edge("comp-avail", "trig-proc", "no")

            if self._sp(sp, "execution"):
                self._add("alloc", "agent", "Allocate Parts",
                          "Inventory reservation · Pick list · Stage", self.CX, self._y)
                self._edge("comp-avail", "alloc", "yes")
                self._adv()
                prev_d = "alloc"

        if self._sp(sp, "execution"):
            self._add("exec", "process", "Release for Repair",
                      "Work order · AR guidance · Checklists", self.CX, self._y)
            prev = next(
                (nid for nid in ("alloc", "comp-avail") if self._find(nid)),
                prev_d
            )
            self._edge(prev, "exec")
            self._adv()

    # ── Quality & certification ───────────────────────────────────────────────
    def _build_quality_certification(self) -> None:
        if not self._sec("quality"):
            return
        sp = "quality"
        prev_q = next(
            (nid for nid in ("exec", "alloc", "reman-gate") if self._find(nid)),
            "reman-hdr"
        )

        if self._sp(sp, "qapass"):
            self._add("qa-pass", "decision", "Quality & Compliance Pass",
                      "Full inspection against standards", self.CX, self._y)
            self._edge(prev_q, "qa-pass")
            self._adv()

            if self._sp(sp, "qaassess"):
                self._add("qa-assess", "quality",
                          "Quality & Compliance Assessment",
                          "Re-assess against compliance criteria",
                          self.CX - 210, self._y - self.ROW_H + 20)
                self._edge("qa-pass",  "qa-assess", "no")
                self._edge("qa-assess","qa-pass")   # loop back
            prev_q = "qa-pass"

        if self._sp(sp, "credit"):
            self._add("credit", "output", "Crediting",
                      "Deposit value calculated & released", self.CX, self._y)
            self._edge(prev_q, "credit", "yes" if self._find("qa-pass") else "main")
            self._adv()
            prev_q = "credit"

        # Report generation side node
        rpt_prev = next(
            (nid for nid in ("credit", "qa-pass", "exec") if self._find(nid)),
            "reman-hdr"
        )
        self._add("rpt-reman", "output", "Report Generation",
                  "Compliance · Summary · Archive",
                  self.CX + 240, self._y - self.ROW_H + 20)
        self._edge(rpt_prev, "rpt-reman")

        if self._sp(sp, "certreport"):
            self._add("cert-rpt", "output", "Certification Report",
                      "Final compliance report issued", self.CX, self._y)
            self._edge(prev_q, "cert-rpt",
                       "yes" if (self._find("qa-pass") and not self._find("credit")) else "main")
            self._adv()

    # ── Knowledge & feedback ──────────────────────────────────────────────────
    def _build_knowledge_feedback(self) -> None:
        if not self._sec("learning"):
            return
        sp = "learning"
        prev_l = next(
            (nid for nid in ("cert-rpt", "credit", "qa-pass") if self._find(nid)),
            "reman-hdr"
        )

        if self._sp(sp, "kb"):
            self._add("kb", "knowledge",
                      "Knowledge Base\n& Training/Feedback loop",
                      "Metadata · Model update · Throughput · Dashboard",
                      self.CX, self._y)
            self._edge(prev_l, "kb")
            self._adv()

        if self._sp(sp, "feedback") and self._find("kb"):
            self._edge("kb", "reman-hdr", "feedback")

    # ── End routing ───────────────────────────────────────────────────────────
    def _build_end_routing(self) -> None:
        defs = {
            "recycle": {"label": "Recycle / Scrap Flow", "desc": "Material value calc · To recycler"},
            "scrap":   {"label": "Scrap Inventory",       "desc": "Material yield tracked"},
            "partial": {"label": "Partial Salvage",        "desc": "Components extracted before scrap"},
            "return":  {"label": "Return to Supplier",     "desc": "Core returned with defect report"},
        }
        scrap_x = self.CX + 440
        sy = (self._find("recycle").y if self._find("recycle") else self._y) + 100
        for rid, active in self.state.end_routing.items():
            if not active or rid == "recycle":  # recycle already placed inline
                continue
            d = defs.get(rid, {"label": rid, "desc": ""})
            self._add(f"end-{rid}", "output", d["label"], d["desc"], scrap_x, sy)
            if self._find("recycle"):
                self._edge("recycle", f"end-{rid}")
            sy += 90


# ═════════════════════════════════════════════════════════════════════════════
# LOGIC ENGINE — validation + enrichment + assembly
# ═════════════════════════════════════════════════════════════════════════════

class LogicEngine:
    """
    Central orchestrator: validates wizard state, constructs the flow,
    enriches agent nodes, and assembles the SMALogicBlock.

    Typical usage
    -------------
    engine = LogicEngine()
    response = engine.build(wizard_state)
    # response.nodes / response.edges / response.logic_block
    """

    BUILDERS = {
        "manufacturing":    ManufacturingFlowBuilder,
        "remanufacturing":  RemanufacturingFlowBuilder,
    }

    def build(self, state: WizardState) -> FlowResponse:
        self._validate(state)
        self._apply_defaults(state)

        builder_cls = self.BUILDERS[state.industry]
        nodes, edges = builder_cls(state).build()

        logic_block = self._assemble(state, nodes, edges)

        return FlowResponse(
            nodes=[n.to_dict() for n in nodes],
            edges=[e.to_dict() for e in edges],
            logic_block=logic_block,
        )

    # ── Validation ────────────────────────────────────────────────────────────
    def _validate(self, state: WizardState) -> None:
        """
        Process: Requirements validation.

        Checks
        ------
        1. industry must be a recognised key
        2. A sub-path marked True under a disabled section is a contradiction
           — we demote it to False with a warning rather than raising an error,
           to keep the UI experience non-blocking.
        3. Numeric targets must be parseable as float (or empty string).
        """
        if state.industry not in self.BUILDERS:
            raise ValueError(
                f"Unknown industry '{state.industry}'. "
                f"Must be one of: {list(self.BUILDERS)}"
            )
        for sec_id, subpaths in state.subpaths.items():
            if state.sections.get(sec_id) is False:
                for path_id in subpaths:
                    subpaths[path_id] = False  # silently demote

        for tid, val in state.targets.items():
            if val and val.strip():
                try:
                    float(val.replace(",", "."))
                except ValueError:
                    raise ValueError(
                        f"Target '{tid}' has non-numeric value '{val}'. "
                        "Targets must be numeric or empty."
                    )

    # ── Apply defaults ────────────────────────────────────────────────────────
    def _apply_defaults(self, state: WizardState) -> None:
        """
        Process: Fill missing wizard keys with domain defaults.

        For each section and sub-path defined in SECTIONS for the chosen
        industry, any key absent from the wizard state is set to its default
        value (True for all standard sub-paths).
        """
        for sec in SECTIONS.get(state.industry, []):
            if sec["id"] not in state.sections:
                state.sections[sec["id"]] = True
            if sec["id"] not in state.subpaths:
                state.subpaths[sec["id"]] = {}
            for sp in sec["subpaths"]:
                if sp["id"] not in state.subpaths[sec["id"]]:
                    state.subpaths[sec["id"]][sp["id"]] = sp["def"]

        for opt in SCRAP_OPTIONS.get(state.industry, []):
            if opt["id"] not in state.end_routing:
                state.end_routing[opt["id"]] = True

    # ── Assemble logic block ──────────────────────────────────────────────────
    def _assemble(self, state: WizardState,
                  nodes: list[Node], edges: list[Edge]) -> dict:
        """
        Process: Logic block assembly.

        Builds the structured SMALogicBlock dict from:
        - Meta (sector, flow_type, description, generated_at, version)
        - Targets (only filled values)
        - Active configuration (sections + subpaths)
        - Flow steps (non-DB nodes sorted by y-position)
        - Decision gates (nodes of type 'decision' with YES/NO routes)
        - Agents (enriched with AgentCatalogue data + ToolCatalogue data)
        - Tools manifest (deduplicated union of all agent tool references)
        - Edges (full list with relation labels)
        """
        edge_list = [e.to_dict() for e in edges]

        filled_targets = {k: v for k, v in state.targets.items() if v}

        flow_steps = [
            {
                "step":        i + 1,
                "node_id":     n.id,
                "node_type":   n.type,
                "label":       n.label,
                "description": n.desc,
                "next": [
                    {"to": e["to"], "condition": None if e["type"] == "main" else e["type"]}
                    for e in edge_list if e["from"] == n.id
                ],
            }
            for i, n in enumerate(
                sorted([n for n in nodes if n.type != "db"], key=lambda n: n.y)
            )
        ]

        decision_gates = [
            {
                "gate_id":     n.id,
                "question":    n.label,
                "description": n.desc,
                "yes_route":   next((e["to"] for e in edge_list if e["from"] == n.id and e["type"] == "yes"), None),
                "no_route":    next((e["to"] for e in edge_list if e["from"] == n.id and e["type"] == "no"),  None),
            }
            for n in nodes if n.type == "decision"
        ]

        agent_nodes = [n for n in nodes if n.type == "agent"]
        enriched_agents = [
            self._enrich_agent(n, edge_list) for n in agent_nodes
        ]

        used_tool_ids = list({
            tid
            for a in enriched_agents
            for tid in ([t["tool_id"] for t in a["tools"]] +
                        [t for sa in a["sub_agents"] for t in sa.get("tools", [])])
        })
        tools_manifest = [
            {"tool_id": tid, **TOOL_CATALOGUE.get(tid, {"type": "external", "description": tid})}
            for tid in used_tool_ids
        ]

        ind = state.industry
        return {
            "sma_logic_block": {
                "meta": {
                    "sector":       ind,
                    "flow_type": (
                        "Closed-loop production optimisation "
                        "(Sub-flow A: Parameter Optimisation + Sub-flow B: Root Cause Analysis)"
                        if ind == "manufacturing" else
                        "Circular economy core recovery "
                        "(Intake → Assessment → R-Strategy → Execution → QA → Knowledge)"
                    ),
                    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "version":      "1.0",
                    "description": (
                        "Self-reinforcing dual-loop system: Sub-flow A optimises parameters "
                        "before and during production; Sub-flow B diagnoses and resolves anomalies. "
                        "Every anomaly resolved by B improves A; every cycle completed by A sharpens B."
                        if ind == "manufacturing" else
                        "End-to-end remanufacturing intelligence: cores are registered, disassembled, "
                        "assessed against COST/TIME/CO2 indicators, routed to the optimal R-strategy, "
                        "repaired, quality-certified, and fed back into the knowledge base."
                    ),
                },
                "targets":           filled_targets,
                "active_sections":   [k for k, v in state.sections.items() if v],
                "active_subpaths": {
                    sec: [k for k, v in paths.items() if v]
                    for sec, paths in state.subpaths.items()
                },
                "end_routing":       [k for k, v in state.end_routing.items() if v],
                "flow_steps":        flow_steps,
                "decision_gates":    decision_gates,
                "agents":            enriched_agents,
                "tools_manifest":    tools_manifest,
                "edges":             edge_list,
            }
        }

    # ── Agent enrichment ──────────────────────────────────────────────────────
    def _enrich_agent(self, node: Node, edge_list: list[dict]) -> dict:
        """
        Process: Agent enrichment.

        Matches the node ID (with fallback to underscore variant) against
        AGENT_CATALOGUE, then resolves each tool ID through TOOL_CATALOGUE.
        Nodes with no catalogue entry fall back to a best-effort enrichment
        derived from the node label and description.
        """
        cat_key = node.id.replace("-", "_")
        cat = AGENT_CATALOGUE.get(node.id) or AGENT_CATALOGUE.get(cat_key)

        in_edges  = [e["from"] for e in edge_list if e["to"]   == node.id]
        out_edges = [{"to": e["to"], "condition": None if e["type"] == "main" else e["type"]}
                     for e in edge_list if e["from"] == node.id]

        if cat:
            tools = [
                {"tool_id": tid, **TOOL_CATALOGUE.get(tid, {"type": "external", "description": tid})}
                for tid in cat["tools"]
            ]
            return {
                "agent_id":     cat_key,
                "label":        node.label,
                "role":         cat["role"],
                "goal":         cat["goal"],
                "instructions": cat["instructions"],
                "triggers_from": in_edges,
                "outputs_to":    out_edges,
                "sub_agents":   cat["sub_agents"],
                "tools":        tools,
            }

        # Fallback: best-effort from node metadata
        return {
            "agent_id":     cat_key,
            "label":        node.label,
            "role":         node.desc,
            "goal":         None,
            "instructions": [s.strip() for s in node.desc.split("·") if s.strip()],
            "triggers_from": in_edges,
            "outputs_to":    out_edges,
            "sub_agents":   [],
            "tools":        [],
        }


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="SMA Logic Engine",
    description=__doc__,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # UI may be opened as file:// or from any dev server
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_engine = LogicEngine()
_html_path = pathlib.Path(__file__).parent / "SMA_Logic_Creator.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui() -> HTMLResponse:
    """Serve the Logic Creator HTML UI."""
    if not _html_path.exists():
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    return HTMLResponse(_html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    """Engine health-check — used by the UI to detect engine availability."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/flow/build")
async def build_flow(
    state: WizardState,
    fmt: str = Query(default="json", alias="format",
                     description="Response format: 'json' or 'yaml'"),
) -> Response:
    """
    Build a flow from wizard state.

    Returns a FlowResponse containing:
    - nodes      : canvas node descriptors (id, type, label, desc, x, y, color, bg, tag)
    - edges      : directed edge descriptors (from, to, type)
    - logic_block: structured SMALogicBlock ready for LLM agent consumption
    """
    try:
        result = _engine.build(state)
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(exc))

    if fmt == "yaml":
        yaml_str = (
            "# SMA Logic Block — Agent-Ready YAML\n"
            f"# Generated: {datetime.datetime.utcnow().isoformat()}Z\n"
            "# Use this file to instruct LLM agents about flow steps, decision gates,\n"
            "# agent roles/goals/instructions, sub-agents, and available tools.\n\n"
        ) + yaml.dump(result.logic_block, allow_unicode=True, sort_keys=False)
        return Response(content=yaml_str, media_type="text/yaml")

    return Response(
        content=json.dumps(result.model_dump(), indent=2),
        media_type="application/json",
    )


@app.get("/api/catalogue/agents")
async def get_agent_catalogue() -> dict:
    """Return the full agent catalogue (role, goal, instructions, sub-agents, tools)."""
    return AGENT_CATALOGUE


@app.get("/api/catalogue/tools")
async def get_tool_catalogue() -> dict:
    """Return the full tool catalogue (type, description)."""
    return TOOL_CATALOGUE


@app.get("/api/catalogue/sections/{industry}")
async def get_sections(industry: str) -> list:
    """Return section + sub-path definitions for the given industry."""
    if industry not in SECTIONS:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown industry: {industry}")
    return SECTIONS[industry]


@app.get("/api/catalogue/targets/{industry}")
async def get_targets(industry: str) -> list:
    """Return target field definitions for the given industry."""
    if industry not in TARGETS:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown industry: {industry}")
    return TARGETS[industry]


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "logic_engine:app",
        host="0.0.0.0",
        port=8765,
        reload=True,
        log_level="info",
    )
