"""Static domain constants for the Claims & Spend Desk world.

Everything that varies per episode (limits, thresholds, people, claims) is drawn
from a seeded RNG in :mod:`claimsdesk.world`; this module only holds the fixed
vocabulary so that every component (generator, tools, rules, rewards) agrees on it.
"""

from __future__ import annotations

YEAR = 2026
COMPANY = "Acme Corp India"

GRADES: dict[str, str] = {
    "G1": "Associate",
    "G2": "Senior Associate",
    "G3": "Manager",
    "G4": "Director",
}
GRADE_ORDER = ["G1", "G2", "G3", "G4"]

DEPARTMENTS = ["Sales", "Engineering", "Finance", "Operations", "Marketing"]

CATEGORIES = ["hotel", "meals", "airfare", "local_transport", "client_entertainment"]

# Domestic cities and their *base* tier. Each world moves two cities between the
# superseded (2024) and current (2026) classification, so the old annex is a
# realistic distractor for retrieval.
DOMESTIC_CITIES: dict[str, int] = {
    "Mumbai": 1, "Delhi": 1, "Bengaluru": 1,
    "Pune": 2, "Hyderabad": 2, "Chennai": 2, "Kolkata": 2, "Ahmedabad": 2,
    "Jaipur": 3, "Kochi": 3, "Indore": 3, "Lucknow": 3, "Nagpur": 3, "Coimbatore": 3,
}
INTERNATIONAL_CITIES = ["Singapore", "London", "New York", "Dubai"]

REGIONS = {
    "Mumbai": "West", "Pune": "West", "Ahmedabad": "West", "Nagpur": "West",
    "Delhi": "North", "Jaipur": "North", "Lucknow": "North", "Indore": "North",
    "Bengaluru": "South", "Hyderabad": "South", "Chennai": "South", "Kochi": "South",
    "Coimbatore": "South", "Kolkata": "East",
}

FX_CURRENCIES = {"USD": 84.0, "EUR": 91.0, "GBP": 106.0, "SGD": 62.5}

DECISIONS = ["approve", "reject", "escalate"]
REASON_CODES = [
    "compliant",
    "duplicate",
    "late_submission",
    "missing_receipt",
    "over_limit",
    "class_not_entitled",
    "needs_senior_approval",
]
DECISION_FOR_REASON = {
    "compliant": "approve",
    "needs_senior_approval": "escalate",
    "duplicate": "reject",
    "late_submission": "reject",
    "missing_receipt": "reject",
    "over_limit": "reject",
    "class_not_entitled": "reject",
}

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna", "Ishaan",
    "Rohan", "Ananya", "Diya", "Aadhya", "Saanvi", "Pari", "Myra", "Anika", "Kavya",
    "Meera", "Riya", "Neha", "Pooja", "Rahul", "Vikram", "Karan", "Siddharth", "Nikhil",
    "Priya", "Sneha", "Tanvi", "Farhan", "Zoya", "Imran", "Ayesha", "Gurpreet", "Harleen",
    "Joseph", "Maria", "Thomas", "Lakshmi", "Venkat", "Suresh", "Divya", "Arvind", "Deepa",
]
LAST_NAMES = [
    "Sharma", "Verma", "Iyer", "Nair", "Reddy", "Patel", "Shah", "Mehta", "Gupta", "Singh",
    "Kapoor", "Joshi", "Kulkarni", "Desai", "Menon", "Pillai", "Rao", "Chatterjee", "Banerjee",
    "Das", "Khan", "Qureshi", "Fernandes", "Dsouza", "Gill", "Bhat", "Kaur", "Agarwal",
    "Mishra", "Pandey", "Naidu", "Krishnan", "Bose", "Sen", "Malhotra", "Chopra",
]
