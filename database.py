import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from config import DB_PATH

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            total_pages INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            file_size INTEGER DEFAULT 0,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Generated Answers and Schemes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generated_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            question TEXT NOT NULL,
            marks INTEGER NOT NULL,
            answer_text TEXT NOT NULL,
            scheme_json TEXT,
            diagram_code TEXT,
            sources_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents (id)
        )
    """)
    
    # Student Evaluations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            answer_id INTEGER,
            question TEXT NOT NULL,
            target_marks REAL NOT NULL,
            marks_awarded REAL NOT NULL,
            student_answer TEXT NOT NULL,
            feedback_json TEXT,
            evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (answer_id) REFERENCES generated_answers (id)
        )
    """)
    
    # Study Bookmarks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            marks INTEGER,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def save_document_record(filename: str, file_path: str, total_pages: int, chunk_count: int, file_size: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (filename, file_path, total_pages, chunk_count, file_size)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, file_path, total_pages, chunk_count, file_size))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def list_documents() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def save_generated_answer(doc_id: Optional[int], question: str, marks: int, answer_text: str, 
                          scheme_dict: Dict, diagram_code: str, sources: List[Dict]) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO generated_answers (doc_id, question, marks, answer_text, scheme_json, diagram_code, sources_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        doc_id,
        question,
        marks,
        answer_text,
        json.dumps(scheme_dict),
        diagram_code,
        json.dumps(sources)
    ))
    ans_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return ans_id

def list_recent_answers(limit: int = 10) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM generated_answers ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = []
    for row in cursor.fetchall():
        d = dict(row)
        if d.get("scheme_json"):
            try:
                d["scheme"] = json.loads(d["scheme_json"])
            except Exception:
                d["scheme"] = {}
        if d.get("sources_json"):
            try:
                d["sources"] = json.loads(d["sources_json"])
            except Exception:
                d["sources"] = []
        rows.append(d)
    conn.close()
    return rows

def save_evaluation(answer_id: Optional[int], question: str, target_marks: float, marks_awarded: float, 
                    student_answer: str, feedback: Dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO student_evaluations (answer_id, question, target_marks, marks_awarded, student_answer, feedback_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        answer_id,
        question,
        target_marks,
        marks_awarded,
        student_answer,
        json.dumps(feedback)
    ))
    eval_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return eval_id

def list_recent_evaluations(limit: int = 10) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM student_evaluations ORDER BY evaluated_at DESC LIMIT ?", (limit,))
    rows = []
    for row in cursor.fetchall():
        d = dict(row)
        if d.get("feedback_json"):
            try:
                d["feedback"] = json.loads(d["feedback_json"])
            except Exception:
                d["feedback"] = {}
        rows.append(d)
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
