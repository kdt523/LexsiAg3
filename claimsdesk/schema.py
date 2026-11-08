"""Logical database schema plus physical *variants*.

The training worlds use variant ``S`` (standard naming). The out-of-distribution
test worlds use variant ``O``: the same data under different table/column names,
so a model that memorised SQL strings instead of reading ``get_schema`` fails.

Gold SQL is written once against the logical schema with ``{table}`` and
``{table.column}`` placeholders and rendered per variant with :func:`render_sql`.
"""

from __future__ import annotations

import re

# logical table -> ordered list of (logical column, sqlite type, description)
LOGICAL_SCHEMA: dict[str, list[tuple[str, str, str]]] = {
    "employees": [
        ("emp_id", "TEXT", "employee id, e.g. E1023"),
        ("name", "TEXT", "full name"),
        ("grade", "TEXT", "G1 Associate, G2 Senior Associate, G3 Manager, G4 Director"),
        ("department", "TEXT", "Sales, Engineering, Finance, Operations or Marketing"),
        ("office_id", "TEXT", "home office id"),
        ("manager_id", "TEXT", "emp_id of line manager (NULL for top level)"),
    ],
    "offices": [
        ("office_id", "TEXT", "office id, e.g. OF03"),
        ("city", "TEXT", "office city"),
        ("region", "TEXT", "North, South, East or West"),
    ],
    "claims": [
        ("claim_id", "TEXT", "claim id, e.g. C-10042"),
        ("emp_id", "TEXT", "claimant employee id"),
        ("category", "TEXT", "hotel, meals, airfare, local_transport or client_entertainment"),
        ("expense_date", "TEXT", "ISO date the expense was incurred (hotel: check-in date)"),
        ("submitted_date", "TEXT", "ISO date the claim was submitted"),
        ("city", "TEXT", "city where the expense was incurred"),
        ("nights", "INTEGER", "hotel nights (hotel claims only, else NULL)"),
        ("flight_hours", "REAL", "flight duration in hours (airfare only, else NULL)"),
        ("cabin_class", "TEXT", "economy or business (airfare only, else NULL)"),
        ("amount", "REAL", "claimed amount in the claim currency"),
        ("currency", "TEXT", "INR or USD"),
        ("receipt_attached", "INTEGER", "1 if a receipt is attached, else 0"),
        ("status", "TEXT", "approved, rejected or pending"),
        ("description", "TEXT", "free-text note from the claimant"),
    ],
    "budgets": [
        ("department", "TEXT", "department name"),
        ("quarter", "TEXT", "e.g. 2026-Q2"),
        ("category", "TEXT", "expense category"),
        ("budget_inr", "REAL", "approved budget in INR"),
    ],
}

VARIANTS: dict[str, dict[str, object]] = {
    "S": {"tables": {}, "columns": {}},
    "O": {
        "tables": {
            "employees": "staff",
            "offices": "locations",
            "claims": "expense_reports",
            "budgets": "cost_plans",
        },
        "columns": {
            "employees": {"emp_id": "staff_no", "name": "full_name", "grade": "band",
                          "department": "dept", "office_id": "location_id",
                          "manager_id": "reports_to"},
            "offices": {"office_id": "location_id", "city": "city_name", "region": "zone"},
            "claims": {"claim_id": "report_id", "emp_id": "staff_no", "category": "expense_type",
                       "expense_date": "incurred_on", "submitted_date": "filed_on",
                       "city": "place", "nights": "num_nights",
                       "flight_hours": "flight_duration_hrs", "cabin_class": "travel_class",
                       "amount": "amount_value", "currency": "currency_code",
                       "receipt_attached": "has_receipt", "status": "state",
                       "description": "memo"},
            "budgets": {"department": "dept", "quarter": "period", "category": "expense_type",
                        "budget_inr": "planned_inr"},
        },
    },
}


class SchemaView:
    """Maps logical table/column names to the physical names of one variant."""

    def __init__(self, variant: str = "S") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown schema variant {variant!r}")
        self.variant = variant
        self._tables: dict[str, str] = VARIANTS[variant]["tables"]  # type: ignore[assignment]
        self._cols: dict[str, dict[str, str]] = VARIANTS[variant]["columns"]  # type: ignore[assignment]

    def t(self, table: str) -> str:
        return self._tables.get(table, table)

    def c(self, table: str, column: str) -> str:
        return self._cols.get(table, {}).get(column, column)

    def ddl(self) -> list[str]:
        stmts = []
        for table, cols in LOGICAL_SCHEMA.items():
            body = ", ".join(f"{self.c(table, c)} {typ}" for c, typ, _ in cols)
            stmts.append(f"CREATE TABLE {self.t(table)} ({body})")
        return stmts

    def describe(self) -> dict[str, list[str]]:
        """Physical schema with short column descriptions (what ``get_schema`` returns)."""
        out: dict[str, list[str]] = {}
        for table, cols in LOGICAL_SCHEMA.items():
            out[self.t(table)] = [f"{self.c(table, c)} {typ} -- {desc}" for c, typ, desc in cols]
        return out


_PLACEHOLDER = re.compile(r"\{([a-z_]+)(?:\.([a-z_]+))?\}")


def render_sql(template: str, view: SchemaView) -> str:
    """Render ``{table}`` / ``{table.column}`` placeholders for a schema variant."""

    def sub(m: re.Match[str]) -> str:
        table, col = m.group(1), m.group(2)
        if table not in LOGICAL_SCHEMA:
            raise KeyError(f"unknown logical table {table!r} in SQL template")
        return view.c(table, col) if col else view.t(table)

    return _PLACEHOLDER.sub(sub, template)
