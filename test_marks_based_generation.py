import sys
import re
from pdf_processor import AcademicPDFProcessor
from rag_engine import AcademicRAGEngine
from generator import AcademicAnswerGenerator, NOT_ENOUGH_INFO_MSG

def run_marks_tests():
    print("=" * 60)
    print("[*] Running Marks-Based Answer Generation Test Suite")
    print("=" * 60)

    # 1. Load OS notes PDF for context
    proc = AcademicPDFProcessor()
    doc = proc.process_file("data/uploads/Sample_Operating_Systems_Notes.pdf")
    rag = AcademicRAGEngine()
    rag.build_index(doc["chunks"])
    doc_id = doc.get("doc_id", doc["chunks"][0].get("doc_id"))
    generator = AcademicAnswerGenerator()

    # ---------------------------------------------------------
    # Test 1: Marks-to-Points Formula Verification (marks * 2)
    # ---------------------------------------------------------
    print("\n[Test 1/5] Testing Marks-to-Points Formula (marks × 2)...")
    test_cases = [
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
        (5, 10),
        (10, 20),
        (15, 30)
    ]
    for marks, expected_pts in test_cases:
        q = f"Explain paging memory management and address translation for {marks} marks."
        chunks = rag.search(q, top_k=5, active_doc_ids=[doc_id])
        res = generator.generate_scheme_and_answer(q, marks=marks, context_chunks=chunks)
        
        assert res["target_points"] == expected_pts, f"Expected {expected_pts} target points for {marks} marks, got {res['target_points']}"
        assert res["marks"] == marks
        assert res["structured_output"]["required_points"] == expected_pts
        print(f"   [PASS] {marks} Mark(s) -> Target Points: {expected_pts} (Generated: {res['generated_points_count']})")

    # ---------------------------------------------------------
    # Test 2: Independent Formula Application (No Total Marks Bleed)
    # ---------------------------------------------------------
    print("\n[Test 2/5] Testing Independent Calculation Per Question...")
    q1 = "Define virtual memory."
    q2 = "Explain the hardware structure of paging."
    q3 = "Discuss the Banker's algorithm data structures."

    res1 = generator.generate_scheme_and_answer(q1, marks=2, context_chunks=rag.search(q1, top_k=3, active_doc_ids=[doc_id]))
    res2 = generator.generate_scheme_and_answer(q2, marks=5, context_chunks=rag.search(q2, top_k=5, active_doc_ids=[doc_id]))
    res3 = generator.generate_scheme_and_answer(q3, marks=10, context_chunks=rag.search(q3, top_k=6, active_doc_ids=[doc_id]))

    assert res1["target_points"] == 4, f"Q1 should have 4 points, got {res1['target_points']}"
    assert res2["target_points"] == 10, f"Q2 should have 10 points, got {res2['target_points']}"
    assert res3["target_points"] == 20, f"Q3 should have 20 points, got {res3['target_points']}"
    print("   [PASS] Q1 (2M=4 pts), Q2 (5M=10 pts), Q3 (10M=20 pts) calculated completely independently.")

    # ---------------------------------------------------------
    # Test 3: Question Type Respect: 'Define' Question
    # ---------------------------------------------------------
    print("\n[Test 3/5] Testing Question Type Adaptation ('Define' -> Definition First)...")
    q_def = "Define the concept of paging in memory management."
    chunks_def = rag.search(q_def, top_k=4, active_doc_ids=[doc_id])
    res_def = generator.generate_scheme_and_answer(q_def, marks=2, context_chunks=chunks_def)
    
    ans_text = res_def["answer_markdown"]
    assert "Question:" in ans_text or "**Question:**" in ans_text
    assert "Marks:" in ans_text or "**Marks:**" in ans_text
    assert "Answer:" in ans_text or "**Answer:**" in ans_text
    # First point should define the term
    lines = [l for l in ans_text.split("\n") if re.match(r'^\d+\.', l.strip())]
    assert len(lines) > 0, "Should generate numbered points"
    first_pt = lines[0].lower()
    assert any(w in first_pt for w in ["paging", "memory", "is a", "refers", "scheme", "management", "concept"]), (
        f"First point should be a definition, got: {first_pt}"
    )
    print(f"   [PASS] Definition positioned as Point 1: '{lines[0][:70]}...'")

    # ---------------------------------------------------------
    # Test 4: Format & Numbering Verification
    # ---------------------------------------------------------
    print("\n[Test 4/5] Testing Output Formatting Structure...")
    q_exp = "Explain the working of address translation in paging."
    chunks_exp = rag.search(q_exp, top_k=5, active_doc_ids=[doc_id])
    res_exp = generator.generate_scheme_and_answer(q_exp, marks=5, context_chunks=chunks_exp)
    
    md = res_exp["answer_markdown"]
    assert "Question:" in md or "**Question:**" in md
    assert "Marks:" in md or "**Marks:**" in md
    assert "Answer:" in md or "**Answer:**" in md
    assert "Points Generated:" in md or "Points Generated" in md

    # Check for duplicate points
    pts = [re.sub(r'^\d+\.\s*', '', l.strip()) for l in md.split("\n") if re.match(r'^\d+\.', l.strip())]
    norm_pts = [re.sub(r'[^a-zA-Z0-9]', '', p.lower()) for p in pts]
    assert len(norm_pts) == len(set(norm_pts)), "Answer points must have zero duplicates"
    print(f"   [PASS] Format strictly adhered to with {len(pts)} unique, numbered points.")

    # ---------------------------------------------------------
    # Test 5: No Hallucination on Absent Content
    # ---------------------------------------------------------
    print("\n[Test 5/5] Testing Grounding Integrity (No Fabrication on Missing Content)...")
    q_absent = "Explain Quantum Entanglement and Shor's Algorithm."
    chunks_absent = rag.search(q_absent, top_k=4, active_doc_ids=[doc_id])
    res_absent = generator.generate_scheme_and_answer(q_absent, marks=10, context_chunks=chunks_absent)
    
    assert res_absent["answer_markdown"] == NOT_ENOUGH_INFO_MSG
    assert res_absent["has_sufficient_context"] is False
    print("   [PASS] Correctly rejected ungrounded question without fabricating filler points.")

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL 5 MARKS-BASED GENERATION TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_marks_tests()
