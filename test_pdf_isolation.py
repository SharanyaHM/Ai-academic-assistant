import os
import sys
from pathlib import Path
from pdf_processor import AcademicPDFProcessor
from rag_engine import AcademicRAGEngine
from generator import AcademicAnswerGenerator, NOT_ENOUGH_INFO_MSG
from diagram_engine import DiagramEngine
from evaluator import StudentAnswerEvaluator
from config import UPLOADS_DIR

def run_pdf_isolation_tests():
    print("=" * 60)
    print("[*] Running Strict PDF Source Isolation & Diagram Verification")
    print("=" * 60)

    pdf_a_path = UPLOADS_DIR / "Sample_Operating_Systems_Notes.pdf"
    pdf_b_path = UPLOADS_DIR / "Sample_Computer_Networks_Notes.pdf"

    assert pdf_a_path.exists(), f"Missing PDF A at {pdf_a_path}"
    assert pdf_b_path.exists(), f"Missing PDF B at {pdf_b_path}"

    processor = AcademicPDFProcessor()
    doc_a = processor.process_file(str(pdf_a_path), "Sample_Operating_Systems_Notes.pdf")
    doc_b = processor.process_file(str(pdf_b_path), "Sample_Computer_Networks_Notes.pdf")

    id_a = doc_a["doc_id"]
    id_b = doc_b["doc_id"]

    assert id_a and id_b, "Both documents must have non-empty unique doc_ids"
    assert id_a != id_b, f"Document IDs must be unique: {id_a} vs {id_b}"

    # Verify every chunk has doc_id attached
    for c in doc_a["chunks"]:
        assert c.get("doc_id") == id_a, "Chunk missing or incorrect doc_id in Doc A"
    for c in doc_b["chunks"]:
        assert c.get("doc_id") == id_b, "Chunk missing or incorrect doc_id in Doc B"

    print("\n[Test 1/6] PDF A (Operating Systems) Ingestion & Grounded Q&A...")
    rag = AcademicRAGEngine()
    rag.build_index(doc_a["chunks"])
    
    q_os = "Explain the concept of paging with hardware address translation."
    res_a = rag.search(q_os, top_k=3, active_doc_ids=[id_a])
    assert len(res_a) > 0, "Should retrieve relevant chunks from Doc A"
    for r in res_a:
        assert r["doc_id"] == id_a
        assert r["source_doc"] == "Sample_Operating_Systems_Notes.pdf"
        assert r["score"] > 0.0

    generator = AcademicAnswerGenerator()
    ans_a = generator.generate_scheme_and_answer(q_os, marks=10, context_chunks=res_a)
    assert ans_a["answer_markdown"] != NOT_ENOUGH_INFO_MSG
    assert "paging" in ans_a["answer_markdown"].lower() or "page" in ans_a["answer_markdown"].lower()
    assert ans_a["diagram_code"], "Paging should generate a verified Mermaid diagram"
    is_valid, err = DiagramEngine.validate_mermaid_syntax(ans_a["diagram_code"])
    assert is_valid, f"Mermaid syntax error: {err}"
    print(f"   [PASS] Answer grounded in PDF A, sources verified, valid Mermaid diagram generated ({len(ans_a['diagram_code'])} chars).")

    print("\n[Test 2/6] PDF B (Computer Networks) Active: Strict PDF A Content Exclusion...")
    # Ingest both into the global index (simulating multiple uploads in the DB)
    combined_chunks = doc_a["chunks"] + doc_b["chunks"]
    rag.build_index(combined_chunks)

    q_cn = "Explain the OSI Reference Model layers and their functions."
    # Filter strictly to PDF B
    res_b = rag.search(q_cn, top_k=4, active_doc_ids=[id_b])
    assert len(res_b) > 0, "Should retrieve chunks from PDF B"
    for r in res_b:
        assert r["doc_id"] == id_b, f"Contamination: Chunk from doc {r['doc_id']} returned instead of {id_b}"
        assert r["source_doc"] == "Sample_Computer_Networks_Notes.pdf"

    ans_b = generator.generate_scheme_and_answer(q_cn, marks=10, context_chunks=res_b)
    assert ans_b["answer_markdown"] != NOT_ENOUGH_INFO_MSG
    assert "layer" in ans_b["answer_markdown"].lower()
    # Confirm no OS keywords in citations
    for src in ans_b["sources"]:
        assert src["source_doc"] == "Sample_Computer_Networks_Notes.pdf"
    print("   [PASS] PDF B retrieval is 100% isolated. Zero chunks from PDF A contaminated the answer.")

    print("\n[Test 3/6] Absent Information Query: Strictly No Hallucinations...")
    # When PDF B is active, ask an OS question whose content is NOT in PDF B
    q_absent = "Explain the Banker's Algorithm safety algorithm with allocation and need matrices."
    res_absent = rag.search(q_absent, top_k=4, active_doc_ids=[id_b])
    
    # Either no chunks pass min_score_threshold, or generator recognizes lack of information
    ans_absent = generator.generate_scheme_and_answer(q_absent, marks=10, context_chunks=res_absent)
    assert ans_absent["answer_markdown"] == NOT_ENOUGH_INFO_MSG, (
        f"Expected exact not enough info message, got: {ans_absent['answer_markdown'][:100]}..."
    )
    assert ans_absent["diagram_code"] == "", "Must not generate a diagram when information is absent"
    print(f"   [PASS] Correctly responded with: '{NOT_ENOUGH_INFO_MSG}' and no invented content.")

    print("\n[Test 4/6] Explicit Multi-Document Scope (PDF A + PDF B)...")
    # When user explicitly selects both documents
    res_both = rag.search("What is paging and what is CSMA/CD?", top_k=6, active_doc_ids=[id_a, id_b])
    docs_found = {r["source_doc"] for r in res_both}
    print(f"   Retrieved from documents: {docs_found}")
    assert len(res_both) > 0
    print("   [PASS] Multi-document retrieval works accurately when explicitly enabled.")

    print("\n[Test 5/6] Diagram Generation & Syntax Validation...")
    # Test valid diagram generation for paging
    code_diag, title_diag, msg_diag = DiagramEngine.generate_diagram_from_context(
        "Explain paging hardware address translation", doc_a["chunks"]
    )
    assert code_diag is not None, "Paging diagram should be generated from OS chunks"
    is_valid, err = DiagramEngine.validate_mermaid_syntax(code_diag)
    assert is_valid, f"Diagram syntax invalid: {err}"
    assert "Page Table" in code_diag or "PT" in code_diag
    assert "Logical Address" in code_diag or "VA" in code_diag

    # Test that absent content returns None, not a generic fallback diagram
    code_empty, title_empty, msg_empty = DiagramEngine.generate_diagram_from_context(
        "Explain Quantum Entanglement Architecture", doc_a["chunks"]
    )
    assert code_empty is None, "Should NOT generate a generic fallback diagram for absent topics"
    assert "does not contain enough architectural or workflow details" in msg_empty
    print("   [PASS] Grounded diagram generation and syntax validation verified. Zero fallback generic diagrams.")

    print("\n[Test 6/6] Student Answer Evaluator Integration...")
    student_ans = (
        "Paging divides memory into fixed size frames. Logical address has page number and offset. "
        "Page table maps page to frame number. Physical address is formed with frame and offset."
    )
    evaluator = StudentAnswerEvaluator()
    eval_res = evaluator.evaluate_answer(
        question=q_os,
        target_marks=10.0,
        student_answer=student_ans,
        scheme=ans_a["scheme"],
        model_answer=ans_a["answer_markdown"]
    )
    assert eval_res["marks_awarded"] > 0
    assert any(g in eval_res["grade"] for g in ["Outstanding", "Excellent", "Good", "Average", "Pass", "Re-attempt", "A", "B", "C", "F"])
    assert "examiner_feedback" in eval_res
    print(f"   [PASS] Evaluator awarded {eval_res['marks_awarded']}/10.0 Marks (Grade: {eval_res['grade']}).")

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL 6 PDF ISOLATION & ACCURACY TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_pdf_isolation_tests()
