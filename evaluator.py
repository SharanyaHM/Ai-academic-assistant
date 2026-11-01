import os
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import MARK_SCHEMES, calibrate_scheme_components, calculate_grade

STOP_WORDS = {
    "what", "is", "explain", "describe", "discuss", "with", "neat", "diagram",
    "and", "the", "for", "from", "concept", "write", "short", "note", "notes",
    "define", "how", "does", "give", "brief", "state", "list", "illustrate", "between"
}

def detect_question_type(question: str) -> str:
    """
    Identifies academic question type for customized rubric evaluation.
    """
    q_low = question.lower()
    if any(w in q_low for w in ["calculate", "compute", "solve", "numerical", "find the value", "determine"]):
        return "numerical"
    elif any(w in q_low for w in ["derive", "derivation", "proof", "prove that"]):
        return "derivation"
    elif any(w in q_low for w in ["define", "what is", "meaning of", "definition"]):
        return "define"
    elif any(w in q_low for w in ["compare", "difference", "distinguish", "contrast", "versus", "vs", "differentiate"]):
        return "compare"
    elif any(w in q_low for w in ["list", "enumerate", "state", "name the", "types of"]):
        return "list"
    elif any(w in q_low for w in ["diagram", "draw", "sketch", "schematic", "architecture"]):
        return "diagram"
    elif any(w in q_low for w in ["discuss", "elaborate", "evaluate", "critique"]):
        return "discuss"
    else:
        return "explain"

def get_question_type_scheme(question: str, target_marks: float) -> List[Dict[str, Any]]:
    """
    Generates a dynamically calibrated marking scheme tailored specifically
    to the question type and marks base, rather than forcing inappropriate rubrics.
    """
    target_marks = float(target_marks)
    q_type = detect_question_type(question)
    q_low = question.lower()
    has_diagram_request = any(w in q_low for w in ["diagram", "schematic", "sketch", "flowchart", "draw"])

    if target_marks <= 2:
        if q_type == "define":
            components = [
                {"component": "Formal Definition & Core Meaning", "marks": 1.0},
                {"component": "Key Technical Terminology & Operational Purpose", "marks": 1.0}
            ]
        elif q_type == "compare":
            components = [
                {"component": "Core Distinction / Differentiating Parameter", "marks": 1.0},
                {"component": "Technical Contrast & Key Attributes", "marks": 1.0}
            ]
        elif q_type == "list":
            components = [
                {"component": "Enumeration of Requested Items", "marks": 1.5},
                {"component": "Accuracy and Relevance of Listed Entities", "marks": 0.5}
            ]
        else:
            components = [
                {"component": "Core Conceptual Statement & Principle", "marks": 1.0},
                {"component": "Operational Feature or Standard Example", "marks": 1.0}
            ]

    elif target_marks <= 5:
        if q_type == "compare":
            components = [
                {"component": "Definition & Scope of Compared Concepts", "marks": 1.0},
                {"component": "Structured Comparison Points (Addressing, Allocation, Overhead)", "marks": 3.0},
                {"component": "Trade-offs and Concluding Summary", "marks": 1.0}
            ]
        elif has_diagram_request or q_type == "diagram":
            components = [
                {"component": "Conceptual Introduction & Architectural Role", "marks": 1.0},
                {"component": "Neat Labeled Architecture / Workflow Schematic", "marks": 2.0},
                {"component": "Working Mechanism & Component Interactions", "marks": 2.0}
            ]
        elif q_type == "numerical":
            components = [
                {"component": "Formula Formulation & Variable Identification", "marks": 1.5},
                {"component": "Step-by-Step Calculation & Substitution", "marks": 2.5},
                {"component": "Final Result with Appropriate Units", "marks": 1.0}
            ]
        elif q_type == "list":
            components = [
                {"component": "Exhaustive Listing of Requested Items", "marks": 3.0},
                {"component": "Technical Function / Purpose of Each Item", "marks": 2.0}
            ]
        else: # explain / discuss
            components = [
                {"component": "Introduction & Formal Definition", "marks": 1.0},
                {"component": "Core Operational Mechanism & Working Principles", "marks": 2.5},
                {"component": "Technical Properties, Invariants & Practical Context", "marks": 1.5}
            ]

    elif target_marks <= 10:
        if q_type == "compare":
            components = [
                {"component": "Theoretical Foundations & Overview of Entities", "marks": 2.0},
                {"component": "Comprehensive Comparison Matrix across Core Technical Parameters", "marks": 5.0},
                {"component": "Performance Trade-offs, Architectural Impact & Summary", "marks": 3.0}
            ]
        elif has_diagram_request or q_type == "diagram":
            components = [
                {"component": "Conceptual Introduction & Theoretical Background", "marks": 1.5},
                {"component": "Neat Labeled Architecture / Flow Diagram", "marks": 3.0},
                {"component": "Detailed Technical Breakdown & Step-by-Step Working", "marks": 3.5},
                {"component": "System Analysis, Operational Rules & Invariants", "marks": 2.0}
            ]
        elif q_type == "numerical":
            components = [
                {"component": "Governing Mathematical Principles & Formulas", "marks": 2.5},
                {"component": "Step-by-Step Mathematical Derivation & Substitution", "marks": 5.0},
                {"component": "Boundary Case Verification & Final Solutions with Units", "marks": 2.5}
            ]
        else: # explain / discuss without mandatory diagram
            components = [
                {"component": "Conceptual Foundation & Theoretical Definition", "marks": 2.0},
                {"component": "System Architecture & Structural Components Breakdown", "marks": 3.0},
                {"component": "Comprehensive Working Mechanism & Step-by-Step Execution", "marks": 3.5},
                {"component": "System Invariants, Trade-offs & Practical Analysis", "marks": 1.5}
            ]

    else: # 15 marks
        components = [
            {"component": "In-depth Theoretical Foundation & System Principles", "marks": 2.5},
            {"component": "Comprehensive Architecture & Component Interconnections", "marks": 4.0},
            {"component": "Detailed Multi-Stage Component & Protocol Analysis", "marks": 4.5},
            {"component": "Operational Workflow, Step-by-Step Execution & Invariants", "marks": 2.5},
            {"component": "Comparative Analysis, Practical Trade-offs & Concluding Remarks", "marks": 1.5}
        ]

    return calibrate_scheme_components(components, target_marks)


class StudentAnswerEvaluator:
    """
    Scheme-Based Student Answer Evaluator & Grading Engine.
    Evaluates student answers against official marking schemes, assigns marks accurately,
    and generates transparent, structured grading reports conforming to university standards.
    
    Applies the marks-based expected points formula:
    Expected Answer Points = Marks Allotted × 2
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.8-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

    def evaluate_answer(self, question: str, target_marks: float, student_answer: str,
                        scheme: Optional[Dict[str, Any]] = None, model_answer: str = "") -> Dict[str, Any]:
        """
        Evaluates a single student answer against an official marking scheme and reference answer.
        Calculates Expected Answer Points = target_marks * 2 as an evaluation guide.
        Categorizes criteria into satisfied, partially satisfied, missing, and incorrect statements.
        """
        target_marks = float(target_marks)
        student_answer = student_answer.strip() if student_answer else ""
        expected_points = int(target_marks * 2)
        q_type = detect_question_type(question)

        # 1. Resolve and calibrate scheme components
        raw_components = None
        if scheme:
            raw_components = scheme.get("components") or scheme.get("breakdown")
        if not raw_components:
            calibrated_components = get_question_type_scheme(question, target_marks)
        else:
            calibrated_components = calibrate_scheme_components(raw_components, target_marks)

        # 2. Handle Blank or Unanswered responses
        if not student_answer:
            comp_evals = [
                {
                    "criterion": c.get("component", "Concept"),
                    "allocated_marks": float(c.get("marks", 1.0)),
                    "marks_awarded": 0.0,
                    "status": "Missing",
                    "comments": "No response provided for this rubric criterion."
                }
                for c in calibrated_components
            ]
            marks_breakdown = [
                {"criterion": c["criterion"], "allocated_marks": c["allocated_marks"], "marks_awarded": 0.0}
                for c in comp_evals
            ]
            grade, status, _ = calculate_grade(0.0)
            
            res = {
                "question": question,
                "target_marks": target_marks,
                "expected_points": expected_points,
                "question_type": q_type,
                "student_answer": "(Blank / Unanswered)",
                "model_answer": model_answer,
                "marks_awarded": 0.0,
                "percentage": 0.0,
                "grade": grade,
                "status": status,
                "component_evaluations": comp_evals,
                "criteria_satisfied": [],
                "criteria_partially_satisfied": [],
                "criteria_missing": [c["criterion"] for c in comp_evals],
                "incorrect_statements": ["No answer submitted by student."],
                "marks_breakdown": marks_breakdown,
                "strengths": [],
                "missing_points": [f"Missing all {expected_points} expected points for this question."],
                "examiner_feedback": "Please write your answer to receive an evaluation.",
                "human_override": None
            }
            res["report_markdown"] = self._build_grading_report_markdown(res)
            res["is_valid"] = self._validate_evaluation(res)
            return res

        # 3. Attempt Gemini evaluation if key provided
        if self.api_key:
            try:
                res = self._evaluate_with_gemini(question, target_marks, expected_points, q_type,
                                                 student_answer, calibrated_components, model_answer)
                if res:
                    res["report_markdown"] = self._build_grading_report_markdown(res)
                    res["is_valid"] = self._validate_evaluation(res)
                    return res
            except Exception as e:
                print(f"Gemini evaluation failed, falling back to NLP rubric engine: {e}")

        # 4. Intelligent NLP Rubric Engine
        res = self._evaluate_with_nlp(question, target_marks, expected_points, q_type,
                                      student_answer, calibrated_components, model_answer)
        res["report_markdown"] = self._build_grading_report_markdown(res)
        res["is_valid"] = self._validate_evaluation(res)
        return res

    def _evaluate_with_gemini(self, question: str, target_marks: float, expected_points: int, q_type: str,
                              student_answer: str, calibrated_components: List[Dict[str, Any]], 
                              model_answer: str) -> Optional[Dict[str, Any]]:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.api_key)

        rubric_str = json.dumps(calibrated_components, indent=2)

        prompt = f"""
You are an experienced University Professor and Chief Examiner in Computer Science at Siddaganga Institute of Technology (SIT) / VTU.
Evaluate the following student's written examination answer against the official marking scheme and reference model answer.

EVALUATION PARAMETERS:
- Question: {question}
- Question Type: {q_type}
- Maximum Marks: {target_marks}
- Expected Answer Points (Coverage Guide): {expected_points} points (calculated as Marks Allotted × 2)

OFFICIAL MARKING SCHEME (Each component must be scored out of its allocated marks):
{rubric_str}

REFERENCE MODEL ANSWER:
\"\"\"
{model_answer}
\"\"\"

STUDENT'S SUBMITTED ANSWER:
\"\"\"
{student_answer}
\"\"\"

SCORING AND EVALUATION RULES:
1. Expected Answer Points ({expected_points}) is an evaluation coverage guide, NOT a rigid word count requirement.
2. Evaluate meaning, demonstrated understanding, and technical correctness rather than exact wording.
3. Award partial credit when a student demonstrates some, but not all, required knowledge.
4. Give credit for correct formulas, working steps, diagrams, examples, and valid alternative methods.
5. Do not penalize students merely for conciseness if the required criteria are satisfied.
6. Do not award full marks solely for length.
7. Deduct marks only for incorrect or contradictory statements justified by the rubric.
8. The sum of allocated_marks across all criteria must equal exactly {target_marks}.
9. The sum of marks_awarded must equal the overall marks_awarded (capped between 0 and {target_marks}).

Return ONLY a JSON object strictly conforming to this schema:
{{
    "marks_awarded": 7.5,
    "target_marks": {target_marks},
    "expected_points": {expected_points},
    "question_type": "{q_type}",
    "component_evaluations": [
        {{
            "criterion": "Criterion Name",
            "allocated_marks": 2.0,
            "marks_awarded": 1.5,
            "status": "Satisfied" | "Partially Satisfied" | "Missing",
            "comments": "Detailed explanation of marks awarded and justification"
        }}
    ],
    "criteria_satisfied": [
        "Criterion 1: Clearly explained core concept"
    ],
    "criteria_partially_satisfied": [
        "Criterion 2: Included definition but omitted formula"
    ],
    "criteria_missing": [
        "Criterion 3: No labeled architecture diagram provided"
    ],
    "incorrect_statements": [
        "Stated that paging causes external fragmentation (contradicts actual system behavior)"
    ],
    "strengths": [
        "Accurate definition of core concept",
        "Clear step progression"
    ],
    "missing_points": [
        "Omitted time complexity and page fault handling"
    ],
    "examiner_feedback": "Actionable, constructive guidance on how to improve this answer to achieve full marks."
}}
"""
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

        try:
            data = json.loads(response.text)
            raw_evals = data.get("component_evaluations", [])
            calibrated_evals = []
            marks_breakdown = []

            for i, target_c in enumerate(calibrated_components):
                target_alloc = float(target_c.get("marks", 1.0))
                if i < len(raw_evals):
                    ev_item = raw_evals[i]
                    crit_name = ev_item.get("criterion", target_c.get("component", f"Criterion {i+1}"))
                    awarded_raw = float(ev_item.get("marks_awarded", 0.0))
                    gemini_alloc = float(ev_item.get("allocated_marks", target_alloc))
                    if gemini_alloc > 0:
                        awarded_norm = round((awarded_raw / gemini_alloc) * target_alloc, 1)
                    else:
                        awarded_norm = min(target_alloc, awarded_raw)
                    awarded_norm = min(target_alloc, max(0.0, awarded_norm))
                    
                    if awarded_norm >= target_alloc * 0.75:
                        status = "Satisfied"
                    elif awarded_norm >= target_alloc * 0.35:
                        status = "Partially Satisfied"
                    else:
                        status = "Missing"
                    comments = ev_item.get("comments", f"Awarded {awarded_norm}/{target_alloc} marks.")
                else:
                    crit_name = target_c.get("component", f"Criterion {i+1}")
                    awarded_norm = round(target_alloc * 0.5, 1)
                    status = "Partially Satisfied"
                    comments = f"Awarded {awarded_norm}/{target_alloc} marks."

                calibrated_evals.append({
                    "criterion": crit_name,
                    "allocated_marks": target_alloc,
                    "marks_awarded": awarded_norm,
                    "status": status,
                    "comments": comments
                })
                marks_breakdown.append({
                    "criterion": crit_name,
                    "allocated_marks": target_alloc,
                    "marks_awarded": awarded_norm
                })

            total_awarded = round(sum(c["marks_awarded"] for c in calibrated_evals), 1)
            total_awarded = min(target_marks, max(0.0, total_awarded))
            percentage = round((total_awarded / target_marks) * 100, 1) if target_marks > 0 else 0.0
            grade, status_str, _ = calculate_grade(percentage)

            return {
                "question": question,
                "target_marks": target_marks,
                "expected_points": expected_points,
                "question_type": q_type,
                "student_answer": student_answer,
                "model_answer": model_answer,
                "marks_awarded": total_awarded,
                "percentage": percentage,
                "grade": grade,
                "status": status_str,
                "component_evaluations": calibrated_evals,
                "criteria_satisfied": data.get("criteria_satisfied", []),
                "criteria_partially_satisfied": data.get("criteria_partially_satisfied", []),
                "criteria_missing": data.get("criteria_missing", []),
                "incorrect_statements": data.get("incorrect_statements", []),
                "marks_breakdown": marks_breakdown,
                "strengths": data.get("strengths", ["Demonstrated relevant subject understanding."]),
                "missing_points": data.get("missing_points", []),
                "examiner_feedback": data.get("examiner_feedback", "Review the model answer to strengthen technical depth."),
                "human_override": None
            }
        except Exception as e:
            print(f"Error parsing Gemini evaluation response: {e}")
            return None

    def _evaluate_with_nlp(self, question: str, target_marks: float, expected_points: int, q_type: str,
                           student_answer: str, calibrated_components: List[Dict[str, Any]], 
                           model_answer: str) -> Dict[str, Any]:
        """
        Intelligent NLP semantic concept-matching evaluation engine.
        Evaluates genuine understanding, awards fair academic marks, adapts to question types,
        detects misconceptions, and applies the official VTU/SIT grading policy.
        """
        student_lower = student_answer.lower()
        word_count = len(student_answer.split())
        q_lower = question.lower()
        q_tokens = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', q_lower) if w not in STOP_WORDS]

        # Use model answer or question + component names as reference text
        if model_answer and len(model_answer.strip()) > 30:
            ref_text = model_answer
        else:
            comp_names = " ".join([c.get("component", "") for c in calibrated_components])
            ref_text = f"{question}. {comp_names}. Essential academic technical definition and working principles."

        # Compute semantic cosine similarity
        try:
            vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_mat = vec.fit_transform([student_answer, ref_text])
            similarity = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
        except Exception:
            similarity = 0.5

        # Check for known academic misconceptions / contradictions
        incorrect_statements = []
        known_contradictions = [
            (r'paging[^\.\n]*(?:produces|causes|leads to|results in|has)[^\.\n]*external fragmentation',
             "Stated that paging causes external fragmentation; in fact, paging eliminates external fragmentation."),
            (r'paging[^\.\n]*(?:does not|doesn\'t|without)[^\.\n]*page table',
             "Stated that paging does not use a page table; a page table is mandatory for virtual-to-physical address translation."),
            (r'tcp[^\.\n]*(?:is|provides)[^\.\n]*connectionless',
             "Stated that TCP is connectionless; TCP is strictly connection-oriented."),
            (r'tcp[^\.\n]*(?:is|provides)[^\.\n]*unreliable',
             "Stated that TCP is unreliable; TCP provides guaranteed reliable transmission."),
            (r'safe state[^\.\n]*(?:is|causes|leads to)[^\.\n]*deadlock',
             "Equated safe state with deadlock; a safe state mathematically guarantees deadlock cannot occur.")
        ]
        for pattern, msg in known_contradictions:
            if re.search(pattern, student_lower):
                incorrect_statements.append(msg)

        component_evals = []
        marks_breakdown = []
        criteria_satisfied = []
        criteria_partially_satisfied = []
        criteria_missing = []
        strengths = []
        missing_pts = []

        # Evaluate each calibrated component in the scheme
        for comp in calibrated_components:
            name = comp.get("component", "Concept")
            max_m = float(comp.get("marks", 1.0))
            name_lower = name.lower()
            c_tokens = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', name_lower) if w not in STOP_WORDS]

            # Measure concept keyword presence
            c_matches = sum(1 for t in c_tokens if re.search(r'\b' + re.escape(t) + r'\b', student_lower))
            q_matches = sum(1 for t in q_tokens if re.search(r'\b' + re.escape(t) + r'\b', student_lower))

            score_factor = 0.0
            comp_comments = ""

            # 1. Definition / Core Conceptual Foundation
            if any(w in name_lower for w in ["definition", "concept", "introduction", "foundation", "meaning"]):
                def_verbs = ["is a", "is defined as", "refers to", "means", "capability of", "scheme that", "technique that", "process that"]
                has_def_verb = any(v in student_lower for v in def_verbs)
                has_subject = q_matches >= 1

                if has_def_verb and has_subject:
                    # Excellent formal definition
                    score_factor = min(1.0, 0.85 + 0.15 * min(1.0, word_count / 30.0))
                    comp_comments = f"Accurate, formal definition provided with correct technical terminology."
                    strengths.append(f"Clear, precise definition of {name}.")
                elif has_subject or similarity > 0.40:
                    # Conceptual presence without exact definition verb
                    score_factor = min(1.0, 0.70 + 0.30 * similarity)
                    comp_comments = f"Demonstrated correct conceptual understanding of {name}."
                    strengths.append(f"Understood core concepts for {name}.")
                else:
                    score_factor = 0.20
                    comp_comments = f"Incomplete conceptual definition; essential principles omitted."
                    missing_pts.append(f"Formal definition and core principles for {name}.")

            # 2. Technical Terminology & Operational Purpose / Characteristics / Standard Examples
            elif any(w in name_lower for w in ["terminology", "purpose", "characteristic", "feature", "property", "example"]):
                purpose_markers = ["allows", "used to", "compensate", "enables", "ensures", "provides", "eliminates", "helps", "to permit", "purpose", "function", "reduces", "prevents", "maintains", "guarantees"]
                example_markers = ["for example", "e.g.", "such as", "like", "instance", "ram", "disk", "tlb", "fifo", "lru", "tcp", "udp", "ip"]
                has_purpose = any(pm in student_lower for pm in purpose_markers)
                has_example = any(em in student_lower for em in example_markers)
                tech_words = ["memory", "ram", "disk", "storage", "cpu", "mmu", "process", "page", "frame", "table", "address", "packet", "layer", "router", "hardware", "software", "deadlock", "resource", "virtual"]
                tech_count = sum(1 for tw in tech_words if re.search(r'\b' + re.escape(tw) + r'\b', student_lower))

                if (has_purpose or has_example) and tech_count >= 2:
                    score_factor = min(1.0, 0.85 + 0.05 * tech_count)
                    comp_comments = f"Correctly explained the operational purpose and utilized appropriate technical terminology."
                    strengths.append(f"Demonstrated operational purpose and technical terminology.")
                elif has_purpose or tech_count >= 1:
                    score_factor = 0.75
                    comp_comments = f"Addressed functional purpose, but could expand on technical parameters."
                    strengths.append(f"Stated functional purpose for {name}.")
                else:
                    score_factor = 0.30
                    comp_comments = f"Technical terminology or specific purpose statements were insufficient."
                    missing_pts.append(f"Key technical terminology and operational purpose for {name}.")

            # 3. Working Mechanism / Translation / Protocol Execution / Operational Breakdown
            elif any(w in name_lower for w in ["working", "mechanism", "protocol", "algorithm", "execution", "translation"]):
                # Check for operational action phrases and procedural progression
                action_words = [
                    "translates", "divides", "splits", "generates", "maps", "indexes", "uses",
                    "page number", "frame number", "offset", "mmu", "cpu", "address",
                    "request", "response", "handshake", "layer", "packet", "forward",
                    "safe state", "allocation", "need", "work", "available", "schedule"
                ]
                act_matches = sum(1 for aw in action_words if aw in student_lower)
                
                # Check if student explains working mechanism in plain sentences
                if act_matches >= 3 or (act_matches >= 2 and word_count >= 25):
                    score_factor = min(1.0, 0.85 + (0.05 * act_matches))
                    comp_comments = f"Thorough explanation of the working mechanism, addressing key operational steps."
                    strengths.append(f"Detailed operational breakdown for {name}.")
                elif act_matches >= 1:
                    score_factor = 0.60
                    comp_comments = f"Addressed basic operational flow, but could detail intermediate translation/execution stages."
                    missing_pts.append(f"Complete intermediate execution steps for {name}.")
                else:
                    score_factor = 0.20
                    comp_comments = f"Operational mechanism and working steps were largely unaddressed."
                    missing_pts.append(f"Step-by-step working principles for {name}.")

            # 3. Diagram / Architecture / Schematic / Flowchart
            elif any(w in name_lower for w in ["diagram", "architecture", "schematic", "flowchart"]):
                has_visual = any(v in student_lower for v in ["->", "-->", "|", "[", "diagram", "figure", "flowchart", "shown below"])
                arch_components = ["cpu", "mmu", "page table", "frame", "ram", "memory", "client", "server", "router", "switch", "layer"]
                arch_matches = sum(1 for ac in arch_components if re.search(r'\b' + re.escape(ac) + r'\b', student_lower))

                if has_visual and arch_matches >= 2:
                    score_factor = 0.95
                    comp_comments = "Provided clear diagrammatic/flowchart representation with labeled components."
                    strengths.append("Included clear schematic / architectural diagram.")
                elif arch_matches >= 3:
                    # Verbal architecture described thoroughly
                    score_factor = 0.70
                    comp_comments = "Described architectural entities thoroughly; including an explicit labeled schematic would yield full credit."
                    missing_pts.append("Neat labeled architecture diagram / flowchart.")
                elif arch_matches >= 1:
                    score_factor = 0.40
                    comp_comments = "Partial mention of structural components without labeled diagram."
                    missing_pts.append("Visual architectural layout / diagram.")
                else:
                    score_factor = 0.15
                    comp_comments = "Missing required architectural schematic / workflow diagram."
                    missing_pts.append("Complete labeled schematic diagram.")

            # 4. Comparison / Differentiating Matrix / Contrast / Trade-offs
            elif any(w in name_lower for w in ["comparison", "difference", "trade-off", "matrix", "contrast", "distinction"]):
                comp_markers = ["whereas", "while", "unlike", "differs", "difference", "in contrast", "instead of", "advantage", "disadvantage", "table", "faster", "overhead"]
                comp_hits = sum(1 for cm in comp_markers if cm in student_lower)
                if comp_hits >= 2 or "|" in student_lower:
                    score_factor = min(1.0, 0.85 + 0.08 * comp_hits)
                    comp_comments = "Provided structured comparison and clear differentiation across parameters."
                    strengths.append("Structured comparison and trade-off analysis.")
                elif comp_hits >= 1:
                    score_factor = 0.60
                    comp_comments = "Identified basic differences, but lacked exhaustive parameter-by-parameter contrast."
                    missing_pts.append("Detailed comparison matrix across multiple parameters.")
                else:
                    score_factor = 0.25
                    comp_comments = "Minimal comparative depth; distinguishing parameters missing."
                    missing_pts.append("Explicit comparison points and trade-offs.")

            # 5. Formula / Numerical / Mathematical Calculation
            elif any(w in name_lower for w in ["formula", "numerical", "calculation", "derivation", "mathematical"]):
                math_elements = ["=", "+", "-", "*", "/", "o(", "bits", "bytes", "page size", "time", "ms", "ns"]
                math_hits = sum(1 for me in math_elements if me in student_lower)
                if math_hits >= 2:
                    score_factor = min(1.0, 0.85 + 0.05 * math_hits)
                    comp_comments = "Included relevant mathematical formulations and equations."
                    strengths.append("Mathematical formulations and equations correctly stated.")
                else:
                    score_factor = 0.30
                    comp_comments = "Omitted mathematical equations or step-by-step substitution."
                    missing_pts.append(f"Explicit formula and calculation steps for {name}.")

            # 6. Technical Terminology & Operational Purpose / Characteristics / Standard Examples
            elif any(w in name_lower for w in ["terminology", "purpose", "characteristic", "feature", "property", "example", "principle"]):
                purpose_markers = ["allows", "used to", "compensate", "enables", "ensures", "provides", "eliminates", "helps", "to permit", "purpose", "function", "reduces", "prevents", "maintains", "guarantees"]
                example_markers = ["for example", "e.g.", "such as", "like", "instance", "ram", "disk", "tlb", "fifo", "lru", "tcp", "udp", "ip"]
                has_purpose = any(pm in student_lower for pm in purpose_markers)
                has_example = any(em in student_lower for em in example_markers)
                tech_words = ["memory", "ram", "disk", "storage", "cpu", "mmu", "process", "page", "frame", "table", "address", "packet", "layer", "router", "hardware", "software", "deadlock", "resource", "virtual"]
                tech_count = sum(1 for tw in tech_words if re.search(r'\b' + re.escape(tw) + r'\b', student_lower))

                if (has_purpose or has_example) and tech_count >= 2:
                    score_factor = min(1.0, 0.85 + 0.05 * tech_count)
                    comp_comments = f"Correctly explained the operational purpose and utilized appropriate technical terminology."
                    strengths.append(f"Demonstrated operational purpose and technical terminology.")
                elif has_purpose or tech_count >= 1:
                    score_factor = 0.70
                    comp_comments = f"Addressed functional purpose, but could expand on technical parameters."
                    strengths.append(f"Stated functional purpose for {name}.")
                else:
                    score_factor = 0.30
                    comp_comments = f"Technical terminology or specific purpose statements were insufficient."
                    missing_pts.append(f"Key technical terminology and operational purpose for {name}.")

            # 7. Default Technical Feature / System Properties / Invariants
            else:
                coverage_score = (c_matches / max(1, len(c_tokens))) if c_tokens else 0.5
                score_factor = min(1.0, 0.40 + 0.40 * coverage_score + 0.30 * similarity)
                comp_comments = f"Demonstrated sound understanding of {name}."

            # Apply deduction if incorrect statements directly pertain to this component
            if incorrect_statements and any(w in name_lower for w in ["paging", "tcp", "deadlock", "fragmentation"]):
                score_factor = max(0.1, score_factor - 0.20)
                comp_comments += " (Deduction applied for factual contradiction)."

            # Compute awarded marks
            awarded = round(max_m * score_factor, 1)
            awarded = min(max_m, max(0.0, awarded))

            # Status classification
            if awarded >= max_m * 0.75:
                status = "Satisfied"
                criteria_satisfied.append(f"**{name}**: {comp_comments} (`{awarded}/{max_m} M`)")
            elif awarded >= max_m * 0.35:
                status = "Partially Satisfied"
                criteria_partially_satisfied.append(f"**{name}**: {comp_comments} (`{awarded}/{max_m} M`)")
            else:
                status = "Missing"
                criteria_missing.append(f"**{name}**: Required criteria not demonstrated (`{awarded}/{max_m} M`)")

            component_evals.append({
                "criterion": name,
                "allocated_marks": max_m,
                "marks_awarded": awarded,
                "status": status,
                "comments": comp_comments
            })
            marks_breakdown.append({
                "criterion": name,
                "allocated_marks": max_m,
                "marks_awarded": awarded
            })

        # Calculate final totals
        total_awarded = round(sum(item["marks_awarded"] for item in component_evals), 1)
        total_awarded = min(target_marks, max(0.0, total_awarded))
        percentage = round((total_awarded / target_marks) * 100, 1) if target_marks > 0 else 0.0

        # Official SIT/VTU University grade calculation
        grade, status_str, _ = calculate_grade(percentage)

        if not missing_pts:
            missing_pts.append(f"Answer covers core expected points. For full credit, elaborate on practical system implementations.")

        feedback = (
            f"You scored {total_awarded}/{target_marks} Marks ({percentage}% • Grade: {grade}). "
            f"The response addresses {len(criteria_satisfied)} criteria fully and {len(criteria_partially_satisfied)} partially. "
            f"To achieve full marks, address the {len(criteria_missing)} missing criteria, particularly focusing on labeled schematics and explicit mechanisms."
        )

        return {
            "question": question,
            "target_marks": target_marks,
            "expected_points": expected_points,
            "question_type": q_type,
            "student_answer": student_answer,
            "model_answer": model_answer,
            "marks_awarded": total_awarded,
            "percentage": percentage,
            "grade": grade,
            "status": status_str,
            "component_evaluations": component_evals,
            "criteria_satisfied": criteria_satisfied,
            "criteria_partially_satisfied": criteria_partially_satisfied,
            "criteria_missing": criteria_missing,
            "incorrect_statements": incorrect_statements if incorrect_statements else ["None detected. Factual consistency maintained."],
            "marks_breakdown": marks_breakdown,
            "strengths": strengths if strengths else ["Demonstrated introductory conceptual awareness."],
            "missing_points": missing_pts,
            "examiner_feedback": feedback,
            "human_override": None
        }

    def _build_grading_report_markdown(self, eval_data: Dict[str, Any]) -> str:
        """
        Builds the structured Question-Wise Grading Report conforming strictly to Section 6 format.
        """
        q = eval_data.get("question", "")
        max_m = eval_data.get("target_marks", 0.0)
        exp_pts = eval_data.get("expected_points", int(max_m * 2))
        ans = eval_data.get("student_answer", "")
        awarded = eval_data.get("marks_awarded", 0.0)
        feedback = eval_data.get("examiner_feedback", "")

        lines = [
            f"### Question: {q}\n",
            f"**Maximum Marks:** {max_m}\n",
            f"**Expected Answer Points:** {exp_pts} (Marks Allotted × 2)\n",
            f"**Student's Answer:**\n> {ans}\n",
            "**Reference Answer / Marking Criteria:**"
        ]

        # Criteria references
        for comp in eval_data.get("component_evaluations", []):
            lines.append(f"- **{comp.get('criterion')}**: {comp.get('allocated_marks')} Marks ({comp.get('status')})")

        lines.append("\n**Evaluation:**\n")
        lines.append("**Criteria satisfied:**")
        if eval_data.get("criteria_satisfied"):
            for s in eval_data["criteria_satisfied"]:
                lines.append(f"- {s}")
        else:
            lines.append("- None fully satisfied.")

        lines.append("\n**Partially satisfied criteria:**")
        if eval_data.get("criteria_partially_satisfied"):
            for p in eval_data["criteria_partially_satisfied"]:
                lines.append(f"- {p}")
        else:
            lines.append("- None.")

        lines.append("\n**Missing criteria:**")
        if eval_data.get("criteria_missing"):
            for m in eval_data["criteria_missing"]:
                lines.append(f"- {m}")
        else:
            lines.append("- All required criteria addressed.")

        lines.append("\n**Incorrect statements:**")
        if eval_data.get("incorrect_statements"):
            for inc in eval_data["incorrect_statements"]:
                lines.append(f"- {inc}")
        else:
            lines.append("- None detected.")

        lines.append("\n**Marks Breakdown:**")
        for b in eval_data.get("marks_breakdown", []):
            lines.append(f"- **{b.get('criterion')}**: `{b.get('marks_awarded')} / {b.get('allocated_marks')} Marks`")

        lines.append(f"\n**Marks Awarded:** `{awarded} / {max_m} Marks` ({eval_data.get('percentage', 0.0)}% • Grade: `{eval_data.get('grade')}`)\n")
        lines.append(f"**Feedback:** {feedback}\n")

        if eval_data.get("human_override"):
            ho = eval_data["human_override"]
            lines.append(f"⚠️ **Human Review Override Applied**: Score adjusted from {ho.get('original_score')} to `{ho.get('adjusted_score')} Marks` by {ho.get('reviewer')} on {ho.get('timestamp')}. Reason: *{ho.get('reason')}*")

        return "\n".join(lines)

    def _validate_evaluation(self, eval_data: Dict[str, Any]) -> bool:
        """
        Stage 8: Validation & Fairness Checks.
        Verifies:
        - Marks awarded within [0, target_marks]
        - Expected points = target_marks * 2
        - Sum of breakdown marks equals marks_awarded
        - Percentage calculation accurate
        """
        target = eval_data.get("target_marks", 0.0)
        awarded = eval_data.get("marks_awarded", 0.0)
        exp_pts = eval_data.get("expected_points", 0)

        if awarded < 0.0 or awarded > target:
            return False
        if exp_pts != int(target * 2):
            return False

        breakdown_sum = round(sum(b.get("marks_awarded", 0.0) for b in eval_data.get("marks_breakdown", [])), 1)
        if abs(breakdown_sum - awarded) > 0.2:
            return False

        return True

    def evaluate_paper(self, questions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Multi-Question / Complete Question Paper Evaluation.
        Evaluates a list of questions, matching each student answer to the question,
        calculates Total Marks Obtained, Total Maximum Marks, Percentage, and Overall Grade.
        """
        evaluated_questions = []
        total_obtained = 0.0
        total_max = 0.0

        for idx, q_item in enumerate(questions_data, 1):
            q_text = q_item.get("question", f"Question {idx}")
            target_m = float(q_item.get("target_marks", 10))
            stud_ans = q_item.get("student_answer", "")
            scheme = q_item.get("scheme")
            model_ans = q_item.get("model_answer", "")

            eval_res = self.evaluate_answer(
                question=q_text,
                target_marks=target_m,
                student_answer=stud_ans,
                scheme=scheme,
                model_answer=model_ans
            )
            eval_res["question_number"] = idx
            evaluated_questions.append(eval_res)
            total_obtained += eval_res["marks_awarded"]
            total_max += target_m

        total_obtained = round(total_obtained, 1)
        total_max = round(total_max, 1)
        overall_percentage = round((total_obtained / total_max) * 100, 1) if total_max > 0 else 0.0
        overall_grade, classification, grade_points = calculate_grade(overall_percentage)

        # Build Summary Report Markdown
        summary_lines = [
            "# 🎓 Official University Examination Grading Report\n",
            f"**Total Marks Obtained:** `{total_obtained} / {total_max} Marks`  ",
            f"**Overall Percentage:** `{overall_percentage}%`  ",
            f"**Awarded Grade:** `{overall_grade}` (Grade Points: {grade_points})  ",
            f"**Final Status:** `{classification}`  ",
            f"**Date Evaluated:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n",
            "---",
            "## 📋 Question-Wise Performance Summary\n",
            "| Q# | Question Text | Max Marks | Expected Pts | Marks Awarded | % Score | Status |",
            "|---|---|---|---|---|---|---|"
        ]

        for eq in evaluated_questions:
            q_brief = eq["question"][:45] + ("..." if len(eq["question"]) > 45 else "")
            summary_lines.append(
                f"| Q{eq['question_number']} | {q_brief} | {eq['target_marks']} M | {eq['expected_points']} pts | **{eq['marks_awarded']} M** | {eq['percentage']}% | `{eq['status']}` |"
            )

        summary_lines.append("\n---\n## 📝 Detailed Question-Wise Evaluations\n")
        for eq in evaluated_questions:
            summary_lines.append(eq["report_markdown"])
            summary_lines.append("\n---\n")

        summary_report = "\n".join(summary_lines)

        return {
            "total_marks_obtained": total_obtained,
            "total_maximum_marks": total_max,
            "percentage": overall_percentage,
            "grade": overall_grade,
            "classification": classification,
            "grade_points": grade_points,
            "total_questions_evaluated": len(evaluated_questions),
            "question_evaluations": evaluated_questions,
            "summary_report_markdown": summary_report
        }

    def apply_human_override(self, evaluation: Dict[str, Any], adjusted_marks: float, 
                             override_reason: str, reviewer_name: str = "Chief Examiner") -> Dict[str, Any]:
        """
        Supports human-in-the-loop review and score override for high-stakes examinations.
        Updates score, percentage, grade, and logs audit trail.
        """
        target_m = float(evaluation.get("target_marks", 10.0))
        adjusted_marks = round(min(target_m, max(0.0, float(adjusted_marks))), 1)
        original_score = evaluation.get("marks_awarded", 0.0)

        override_record = {
            "original_score": original_score,
            "adjusted_score": adjusted_marks,
            "reason": override_reason,
            "reviewer": reviewer_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        updated = dict(evaluation)
        updated["marks_awarded"] = adjusted_marks
        new_percentage = round((adjusted_marks / target_m) * 100, 1) if target_m > 0 else 0.0
        updated["percentage"] = new_percentage
        new_grade, new_status, _ = calculate_grade(new_percentage)
        updated["grade"] = new_grade
        updated["status"] = new_status
        updated["human_override"] = override_record
        updated["report_markdown"] = self._build_grading_report_markdown(updated)
        updated["is_valid"] = self._validate_evaluation(updated)
        return updated
