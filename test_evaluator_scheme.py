import sys
from evaluator import StudentAnswerEvaluator, detect_question_type
from config import calculate_grade

def run_evaluator_tests():
    print("=" * 60)
    print("[*] Running Scheme-Based Student Answer Evaluator Test Suite")
    print("=" * 60)

    evaluator = StudentAnswerEvaluator()

    # ---------------------------------------------------------
    # Test 1: Expected Points Formula (Marks × 2)
    # ---------------------------------------------------------
    print("\n[Test 1/6] Testing Expected Points Formula (Marks × 2 Guide)...")
    for m in [1, 2, 3, 5, 10, 15]:
        res = evaluator.evaluate_answer(
            question="Explain concept",
            target_marks=m,
            student_answer="Paging divides memory into fixed pages and physical frames.",
            scheme=None
        )
        assert res["expected_points"] == m * 2, f"Expected {m*2} points, got {res['expected_points']}"
        assert res["target_marks"] == float(m)
        print(f"   [PASS] {m} Mark(s) -> Expected Points: {res['expected_points']} (Guide)")

    # ---------------------------------------------------------
    # Test 2: Blank / Unanswered Response
    # ---------------------------------------------------------
    print("\n[Test 2/6] Testing Blank / Unanswered Response Handling...")
    blank_res = evaluator.evaluate_answer(
        question="Explain paging",
        target_marks=10,
        student_answer="",
        scheme=None
    )
    assert blank_res["marks_awarded"] == 0.0
    assert blank_res["percentage"] == 0.0
    assert blank_res["status"] == "Fail"
    assert len(blank_res["criteria_missing"]) > 0
    assert "report_markdown" in blank_res
    print("   [PASS] Blank answer correctly awarded 0.0 Marks with complete missing criteria list.")

    # ---------------------------------------------------------
    # Test 3: Fully Correct vs Partially Correct vs Misconception
    # ---------------------------------------------------------
    print("\n[Test 3/6] Testing Correct, Partial, and Misconception Handling...")
    q = "Explain the concept of paging with hardware address translation."
    target_m = 10
    
    # 3a. Good student answer
    good_ans = (
        "Paging is a memory management scheme that eliminates the need for contiguous allocation of physical memory. "
        "The CPU generates a logical address divided into Page Number (p) and Page Offset (d). "
        "The MMU uses the page table to translate page number p to frame number f. "
        "Physical address is calculated as base frame address + offset d. "
        "The architecture consists of CPU, MMU, Page Table in memory, and Physical RAM frames shown in the flowchart."
    )
    res_good = evaluator.evaluate_answer(q, target_m, good_ans)
    assert res_good["marks_awarded"] >= 6.0, f"Good answer should receive >= 6.0 marks, got {res_good['marks_awarded']}"
    assert len(res_good["criteria_satisfied"]) > 0
    print(f"   [PASS] Strong answer scored {res_good['marks_awarded']}/{target_m} M (Grade: {res_good['grade']}).")

    # 3b. Partial student answer
    partial_ans = "Paging is a memory management technique in operating systems."
    res_partial = evaluator.evaluate_answer(q, target_m, partial_ans)
    assert res_partial["marks_awarded"] < res_good["marks_awarded"], "Partial answer should score less than thorough answer"
    assert len(res_partial["criteria_missing"]) > 0 or len(res_partial["criteria_partially_satisfied"]) > 0
    print(f"   [PASS] Partial answer scored {res_partial['marks_awarded']}/{target_m} M with missing criteria flagged.")

    # 3c. Misconception / Contradictory statement
    misconception_ans = (
        "Paging is used in OS. However, paging produces high external fragmentation and does not use a page table."
    )
    res_misc = evaluator.evaluate_answer(q, target_m, misconception_ans)
    assert any("contradict" in inc.lower() or "external fragmentation" in inc.lower() for inc in res_misc["incorrect_statements"])
    print(f"   [PASS] Contradictory statement detected: '{res_misc['incorrect_statements'][0][:75]}...'")

    # 3d. Textbook 2M definition
    def_ans = "Virtual memory is a memory management capability of an operating system that allows a computer to compensate for physical memory shortages by temporarily transferring data from RAM to disk storage."
    res_def = evaluator.evaluate_answer("Define virtual memory.", 2, def_ans)
    assert res_def["marks_awarded"] >= 1.8, f"Textbook definition should receive >= 1.8/2.0 marks, got {res_def['marks_awarded']}"
    assert res_def["grade"].startswith("O") or res_def["grade"].startswith("A")
    print(f"   [PASS] 2M definition answer scored {res_def['marks_awarded']}/2.0 M (Grade: {res_def['grade']}).")

    # ---------------------------------------------------------
    # Test 4: Question Type Specific Detection & Evaluation
    # ---------------------------------------------------------
    print("\n[Test 4/6] Testing Question-Type Detection...")
    assert detect_question_type("Define virtual memory") == "define"
    assert detect_question_type("Compare paging and segmentation") == "compare"
    assert detect_question_type("List the functions of transport layer") == "list"
    assert detect_question_type("Calculate effective memory access time with TLB") == "numerical"
    assert detect_question_type("Draw the labeled architecture of OSI model") == "diagram"
    assert detect_question_type("Discuss Banker's deadlock avoidance algorithm") == "discuss"
    assert detect_question_type("Explain process lifecycle") == "explain"
    print("   [PASS] Question types accurately classified.")

    # ---------------------------------------------------------
    # Test 5: Standard Grading Report Formatting (Section 6)
    # ---------------------------------------------------------
    print("\n[Test 5/6] Testing Section 6 Standard Grading Report Format...")
    report = res_good["report_markdown"]
    assert "Question:" in report
    assert "Maximum Marks:" in report
    assert "Expected Answer Points:" in report
    assert "Student's Answer:" in report
    assert "Reference Answer / Marking Criteria:" in report
    assert "Evaluation:" in report
    assert "Criteria satisfied:" in report
    assert "Marks Breakdown:" in report
    assert "Marks Awarded:" in report
    assert "Feedback:" in report
    print("   [PASS] Grading report strictly conforms to Section 6 format.")

    # ---------------------------------------------------------
    # Test 6: Multi-Question Paper Evaluation & Human Override
    # ---------------------------------------------------------
    print("\n[Test 6/6] Testing Multi-Question Paper Evaluation & Human Override...")
    paper = [
        {"question": "Define paging.", "target_marks": 2, "student_answer": "Paging is a memory management scheme that divides logical address into pages and physical memory into frames."},
        {"question": "Explain OSI Layer 3.", "target_marks": 5, "student_answer": "Network layer handles logical routing, IP addressing, and packet forwarding across heterogeneous networks."},
        {"question": "Discuss Banker's Algorithm.", "target_marks": 10, "student_answer": ""} # blank
    ]
    paper_res = evaluator.evaluate_paper(paper)
    assert paper_res["total_maximum_marks"] == 17.0
    assert paper_res["total_marks_obtained"] > 0
    assert paper_res["total_questions_evaluated"] == 3
    assert len(paper_res["question_evaluations"]) == 3
    assert paper_res["percentage"] == round((paper_res["total_marks_obtained"] / 17.0) * 100, 1)
    print(f"   [PASS] Paper Evaluation: {paper_res['total_marks_obtained']}/17.0 M ({paper_res['percentage']}%) - Grade: {paper_res['grade']}.")

    # Test Human Override
    overridden = evaluator.apply_human_override(
        res_partial, 
        adjusted_marks=4.0, 
        override_reason="Awarded discretionary marks for student handwriting and valid partial diagram.",
        reviewer_name="Prof. Sharma"
    )
    assert overridden["marks_awarded"] == 4.0
    assert overridden["human_override"]["adjusted_score"] == 4.0
    assert overridden["human_override"]["reviewer"] == "Prof. Sharma"
    assert "Human Review Override Applied" in overridden["report_markdown"]
    print("   [PASS] Human override and audit trail successfully logged.")

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL 6 EVALUATOR & GRADING TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluator_tests()
