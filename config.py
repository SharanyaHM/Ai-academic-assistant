import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "academic_assistant.db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Project Information
PROJECT_INFO = {
    "title": "AI Academic Assistant: Intelligent PDF Analysis and Scheme-Based Answer Generation"
}

# Marking Scheme Specifications according to SIT/VTU guidelines
MARK_SCHEMES = {
    2: {
        "label": "2 Marks (Short Answer)",
        "expected_length": "30 - 60 words (3-5 sentences / bullet points)",
        "breakdown": [
            {"component": "Crisp Definition / Statement / Principle", "marks": 1.0},
            {"component": "Key Feature, Formula, or Standard Example", "marks": 1.0}
        ],
        "components": [
            {"component": "Crisp Definition / Statement / Principle", "marks": 1.0},
            {"component": "Key Feature, Formula, or Standard Example", "marks": 1.0}
        ],
        "diagram_required": False,
        "description": "Precise, conceptual definition or core equation without unnecessary filler."
    },
    5: {
        "label": "5 Marks (Medium Answer)",
        "expected_length": "120 - 200 words (structured paragraphs or 4-6 points)",
        "breakdown": [
            {"component": "Introduction & Definition", "marks": 1.0},
            {"component": "Core Explanation / Working Principle (3-4 points)", "marks": 2.5},
            {"component": "Schematic / Syntax / Formula / Example", "marks": 1.5}
        ],
        "components": [
            {"component": "Introduction & Definition", "marks": 1.0},
            {"component": "Core Explanation / Working Principle (3-4 points)", "marks": 2.5},
            {"component": "Schematic / Syntax / Formula / Example", "marks": 1.5}
        ],
        "diagram_required": "Optional / Recommended if architectural or schematic",
        "description": "Structured explanation covering fundamentals, working, and a relevant example or simple block diagram."
    },
    10: {
        "label": "10 Marks (Long Answer)",
        "expected_length": "350 - 550 words (comprehensive technical essay)",
        "breakdown": [
            {"component": "Introduction & Conceptual Background", "marks": 1.5},
            {"component": "Neat Labeled Diagram / Architecture / Flowchart", "marks": 3.0},
            {"component": "Detailed Working / Algorithm / Derivation / Mechanism", "marks": 3.5},
            {"component": "Real-world Example, Pros/Cons or Complexity Analysis", "marks": 2.0}
        ],
        "components": [
            {"component": "Introduction & Conceptual Background", "marks": 1.5},
            {"component": "Neat Labeled Diagram / Architecture / Flowchart", "marks": 3.0},
            {"component": "Detailed Working / Algorithm / Derivation / Mechanism", "marks": 3.5},
            {"component": "Real-world Example, Pros/Cons or Complexity Analysis", "marks": 2.0}
        ],
        "diagram_required": True,
        "description": "Comprehensive academic answer with neat labeled diagram, in-depth technical explanation, and analytical insights."
    },
    15: {
        "label": "15 Marks (Comprehensive / Essay Answer)",
        "expected_length": "600 - 900 words (thorough multi-section treatise)",
        "breakdown": [
            {"component": "Introduction & Theoretical Foundation", "marks": 2.0},
            {"component": "Comprehensive Architecture & Labeled Diagram", "marks": 4.0},
            {"component": "In-depth Component Analysis & Operation Workflow", "marks": 5.0},
            {"component": "Mathematical Formulation / Algorithm & Code / Case Study", "marks": 2.5},
            {"component": "Comparison Table, Advantages, Limitations & Summary", "marks": 1.5}
        ],
        "components": [
            {"component": "Introduction & Theoretical Foundation", "marks": 2.0},
            {"component": "Comprehensive Architecture & Labeled Diagram", "marks": 4.0},
            {"component": "In-depth Component Analysis & Operation Workflow", "marks": 5.0},
            {"component": "Mathematical Formulation / Algorithm & Code / Case Study", "marks": 2.5},
            {"component": "Comparison Table, Advantages, Limitations & Summary", "marks": 1.5}
        ],
        "diagram_required": True,
        "description": "Exhaustive multi-part answer for full-module coverage, rigorous diagrams, algorithms, and comparative analysis."
    }
}

def calibrate_scheme_components(components_list: list, target_marks: float) -> list:
    """
    Calibrates component allocated marks proportionally so they strictly divide
    and sum up to target_marks exactly.
    """
    if not components_list:
        return []

    target_marks = float(target_marks)
    current_marks = []
    for c in components_list:
        try:
            m = float(c.get("marks", 1.0))
        except (ValueError, TypeError):
            m = 1.0
        current_marks.append(max(0.1, m))

    total_current = sum(current_marks)
    if total_current <= 0:
        total_current = float(len(components_list))
        current_marks = [1.0] * len(components_list)

    # Scale proportionally
    scaled_marks = [round((m / total_current) * target_marks, 1) for m in current_marks]

    # Adjust rounding remainder on the largest element
    diff = round(target_marks - sum(scaled_marks), 1)
    if diff != 0:
        max_idx = scaled_marks.index(max(scaled_marks))
        scaled_marks[max_idx] = round(scaled_marks[max_idx] + diff, 1)

    calibrated = []
    for c, sm in zip(components_list, scaled_marks):
        comp_copy = dict(c)
        comp_copy["marks"] = sm
        calibrated.append(comp_copy)

    return calibrated

# Stable Gemini models suitable for new API projects. Labels are shown in the
# UI; values are the model IDs sent to the Gemini API.
GEMINI_MODELS = {
    "Gemini 3.8 Flash — Recommended": "gemini-3.8-flash",
    "Gemini 3.6 Flash — Balanced": "gemini-3.6-flash",
    "Gemini 3.5 Flash-Lite — Faster, lower cost": "gemini-3.5-flash-lite",
}

# SIT/VTU University Configured Grading Scale
GRADING_SCALE = [
    {"min_percent": 90.0, "grade": "O (Outstanding)", "points": 10, "status": "Pass"},
    {"min_percent": 80.0, "grade": "A+ (Excellent)", "points": 9, "status": "Pass"},
    {"min_percent": 70.0, "grade": "A (Very Good)", "points": 8, "status": "Pass"},
    {"min_percent": 60.0, "grade": "B+ (Good)", "points": 7, "status": "Pass"},
    {"min_percent": 50.0, "grade": "B (Above Average)", "points": 6, "status": "Pass"},
    {"min_percent": 40.0, "grade": "C (Pass)", "points": 5, "status": "Pass"},
    {"min_percent": 0.0, "grade": "F (Re-attempt Recommended)", "points": 0, "status": "Fail"}
]

def calculate_grade(percentage: float) -> tuple:
    """
    Returns (grade_string, status_string, grade_points) according to SIT/VTU guidelines.
    """
    for level in GRADING_SCALE:
        if percentage >= level["min_percent"]:
            return level["grade"], level["status"], level["points"]
    return "F (Re-attempt Recommended)", "Fail", 0
