import os
from pathlib import Path
from pdf_processor import AcademicPDFProcessor
from rag_engine import AcademicRAGEngine
from generator import AcademicAnswerGenerator
from evaluator import StudentAnswerEvaluator
from database import init_db, list_documents, list_recent_answers, list_recent_evaluations
from config import UPLOADS_DIR

def run_tests():
    print("==================================================")
    print("[*] Running AI Academic Assistant System Verification")
    print("==================================================")

    # 1. Database Init
    print("\n[1/5] Testing Database Initialization...")
    init_db()
    print("[PASS] Database initialized successfully.")

    # 2. PDF Processing
    print("\n[2/5] Testing PDF Processing...")
    processor = AcademicPDFProcessor()
    sample_pdf = UPLOADS_DIR / "Sample_Operating_Systems_Notes.pdf"
    assert sample_pdf.exists(), f"Sample PDF missing at {sample_pdf}"
    
    doc_info = processor.process_file(str(sample_pdf))
    print(f"[PASS] Extracted {doc_info['total_pages']} pages and {doc_info['total_chunks']} chunks from {sample_pdf.name}")
    assert doc_info["total_pages"] > 0
    assert doc_info["total_chunks"] > 0

    # 3. RAG Semantic Retrieval
    print("\n[3/5] Testing RAG Semantic Retrieval Engine...")
    rag = AcademicRAGEngine()
    rag.build_index(doc_info["chunks"])
    results = rag.search("Explain paging address translation", top_k=3)
    print(f"[PASS] Retrieved {len(results)} chunks for query 'Explain paging address translation'")
    for r in results:
        print(f"   - Page {r['page']} | Score: {r['score']} | {r['text'][:60]}...")
    assert len(results) > 0

    # 4. Answer & Scheme Generation
    print("\n[4/5] Testing 10-Mark Answer, Scheme & Diagram Generation...")
    generator = AcademicAnswerGenerator()
    q = "Explain the concept of paging with hardware address translation and neat diagram."
    gen_result = generator.generate_scheme_and_answer(q, marks=10, context_chunks=results)
    
    print(f"[PASS] Generated answer for: '{gen_result['question']}' ({gen_result['marks']} Marks)")
    print(f"   - Scheme Components: {len(gen_result['scheme']['components'])}")
    for c in gen_result['scheme']['components']:
        print(f"     * {c['component']}: {c['marks']} Marks")
    print(f"   - Diagram Needed: {gen_result['diagram_needed']}")
    print(f"   - Diagram Code Length: {len(gen_result.get('diagram_code', ''))} chars")
    assert gen_result["marks"] == 10
    assert len(gen_result["scheme"]["components"]) > 0

    # 5. Student Answer Evaluation
    print("\n[5/5] Testing Student Answer Evaluation...")
    student_ans = (
        "Paging is a memory management scheme that avoids contiguous allocation. "
        "The CPU generates a logical address containing page number and offset. "
        "The page table maps the page number to frame number. "
        "Physical address is calculated by frame number and offset. "
        "TLB is used to speed up the process."
    )
    evaluator = StudentAnswerEvaluator()
    eval_res = evaluator.evaluate_answer(
        question=q,
        target_marks=10.0,
        student_answer=student_ans,
        scheme=gen_result["scheme"],
        model_answer=gen_result["answer_markdown"]
    )
    print(f"[PASS] Evaluation Result:")
    print(f"   - Marks Awarded: {eval_res['marks_awarded']} / {eval_res['target_marks']}")
    print(f"   - Percentage: {eval_res['percentage']}% | Grade: {eval_res['grade']}")
    print(f"   - Missing Points Count: {len(eval_res['missing_points'])}")
    print(f"   - Examiner Feedback: {eval_res['examiner_feedback'][:100]}...")
    assert eval_res["marks_awarded"] > 0

    print("\n==================================================")
    print("[SUCCESS] ALL 5 SYSTEM TEST SUITES PASSED FLAWLESSLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
