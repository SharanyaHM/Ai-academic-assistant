import os
import json
import re
import time
from typing import List, Dict, Any, Optional
from config import MARK_SCHEMES, calibrate_scheme_components
from diagram_engine import DiagramEngine
from structured_output import StructuredOutputPipeline

NOT_ENOUGH_INFO_MSG = "The uploaded PDF does not contain enough relevant information to answer this question."

# Avoid repeating long Gemini requests while the service reports overload or
# rate limiting. Local, document-grounded synthesis remains available meanwhile.
_GEMINI_RETRY_AFTER = 0.0
_GEMINI_COOLDOWN_SECONDS = 60.0


def _clean_answer_text(text: str) -> str:
    """Remove PDF/model encoding debris without changing ordinary punctuation."""
    text = str(text or "")
    replacements = {
        "\ufffd": " ", "\u0000": " ", "â€™": "'", "â€˜": "'",
        "â€œ": '"', "â€": '"', "â€“": "–", "â€”": "—", "â€¦": "…",
        "Â ": " ", "Â": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()


def _format_answer_markdown(answer: str, question: str, marks: int,
                            generated_count: int, target_points: int) -> str:
    """Normalize model output into one readable, sequential point per Markdown block."""
    answer = _clean_answer_text(answer)
    # Model output may contain the header already; rebuild it consistently.
    body = re.sub(r"(?im)^\s*\*{0,2}(?:question|marks):?\*{0,2}\s*:?[ \t]*[^\n]*(?:\n|$)", "", answer)
    body = re.sub(r"(?im)^\s*\*{0,2}answer:?\*{0,2}\s*:?[ \t]*", "", body)
    body = re.sub(r"(?m)^\s*(?:📌\s*)?\*{0,2}Points Generated\*{0,2}.*$", "", body)
    body = body.strip()

    # Number markers are often collapsed into a single paragraph by model output.
    # Split only at existing numbered markers; never split sentences to meet a quota.
    markers = list(re.finditer(r"(?:^|\s)(\d{1,3})[.)]\s+", body))
    points = []
    leading = ""
    if markers:
        leading = body[:markers[0].start()].strip(" \n:-")
        for index, marker in enumerate(markers):
            start = marker.end()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(body)
            content = re.sub(r"\s+", " ", body[start:end]).strip(" \n")
            content = re.sub(r"\s*(?:\*{0,2})?(?:Points Generated|Sources?):.*$", "", content, flags=re.I).strip()
            if content:
                points.append(content)

    lines = [f"**Question:** {question}", "", f"**Marks:** {marks}", "", "**Answer:**"]
    if leading:
        lines.extend(["", f"**Introduction:** {leading}"])
    if points:
        for number, point in enumerate(points, 1):
            # Keep the point once, without deriving a bold heading from its
            # opening words (which repeated and visually over-emphasized them).
            point = re.sub(r"\*\*(.*?)\*\*", r"\1", point)
            lines.extend(["", f"{number}. {point}"])
    elif body:
        # Preserve unnumbered/table responses (notably existing comparison tables).
        lines.extend(["", body])
    display_count = len(points) if points else generated_count
    lines.extend(["", f"📌 **Points Generated:** {display_count} / {target_points} required ({marks} marks × 2)"])
    return "\n".join(lines)

STOP_WORDS = {
    "what", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "this", "these", "those", "there", "here", "some", "any", "which", "that", "their",
    "can", "will", "would", "should", "could", "also", "than", "then", "when", "where",
    "explain", "describe", "discuss", "with", "neat", "diagram", "and", "the", "for",
    "from", "concept", "write", "short", "note", "notes",
    "define", "how", "does", "give", "brief", "state", "list", "illustrate",
    "working", "mechanism", "using", "based", "marks", "mark", "explain", "key"
}

class AcademicAnswerGenerator:
    """
    Intelligent Answer and Marking Scheme Generator.
    Produces exam-ready, mark-calibrated answers with transparent scoring rubrics
    and labeled architectural/workflow diagrams, strictly grounded in uploaded PDF documents.
    Transforms raw extracted text into structured, typed, user-friendly responses.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.8-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

    def generate_scheme_and_answer(self, question: str, marks: int, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for generating mark-appropriate answer, marking scheme rubric,
        and conceptual diagram strictly grounded in the provided context_chunks.
        """
        # If no context chunks were retrieved, immediately indicate lack of information
        if not context_chunks:
            return {
                "question": question,
                "marks": marks,
                "scheme": {
                    "target_marks": marks,
                    "components": [
                        {"component": "Relevance & Content Verification", "marks": marks}
                    ],
                    "total_allocated": marks
                },
                "answer_markdown": NOT_ENOUGH_INFO_MSG,
                "structured_output": {
                    "schema_version": "2.0",
                    "topic": question,
                    "content_type": "conceptual_definition",
                    "domain": "Academic Repository",
                    "executive_summary": NOT_ENOUGH_INFO_MSG,
                    "key_entities": [],
                    "sections": [],
                    "technical_attributes": {},
                    "missing_or_uncertain_fields": ["All required content is absent from the selected document."],
                    "examiner_takeaways": ["Verify that your question is covered in the active document."]
                },
                "is_structured": False,
                "diagram_code": "",
                "diagram_needed": False,
                "diagram_title": "",
                "diagram_message": "No diagram generated because relevant content was not found in the uploaded document.",
                "tips_for_full_marks": [
                    "Ensure the question is directly covered in the uploaded course materials."
                ],
                "sources": [],
                "has_sufficient_context": False
            }

        scheme_template = MARK_SCHEMES.get(marks, MARK_SCHEMES[10])
        diagram_needed = DiagramEngine.should_include_diagram(question, marks, context_chunks=context_chunks)

        # Context compilation from RAG chunks
        context_text = "\n\n".join([
            f"[Source: {c.get('source_doc', 'Document')} | Page {c.get('page', 1)}]\n{c['text']}"
            for c in context_chunks
        ])

        # Attempt Gemini generation if configured, but avoid repeatedly waiting
        # on a service that recently returned a transient overload/quota error.
        global _GEMINI_RETRY_AFTER
        if self.api_key and time.monotonic() >= _GEMINI_RETRY_AFTER:
            try:
                result = self._generate_with_gemini(question, marks, context_text, scheme_template, diagram_needed)
                if result:
                    result["sources"] = context_chunks
                    return result
            except Exception as e:
                print(f"Gemini API call failed, falling back to academic engine: {e}")
                error_text = str(e).upper()
                if "503" in error_text or "UNAVAILABLE" in error_text or "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                    _GEMINI_RETRY_AFTER = time.monotonic() + _GEMINI_COOLDOWN_SECONDS
        elif self.api_key:
            print("Gemini temporarily paused after a recent overload/rate-limit error; using the local academic engine.")

        # Grounded Academic Knowledge Synthesis Engine
        return self._generate_with_synthesis_engine(question, marks, context_chunks, scheme_template, diagram_needed)

    def _generate_with_gemini(self, question: str, marks: int, context_text: str, 
                              scheme_template: Dict, diagram_needed: bool) -> Optional[Dict[str, Any]]:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.api_key)

        system_prompt = f"""
You are an expert Academic Professor and Chief Paper Evaluator in Computer Science and Engineering at Siddaganga Institute of Technology (SIT) / VTU.
Your task is to transform academic source content into a validated, highly structured academic response strictly tailored to {marks} Marks.

CRITICAL INSTRUCTIONS:
1. Treat the provided CONTEXT as raw source data to analyze, NOT as the final response.
2. Analyze the context to extract key entities, definitions, structural components, operational steps, and technical attributes.
3. If the context does not contain enough relevant information to answer the question, return:
   "answer_markdown": "{NOT_ENOUGH_INFO_MSG}"
   "diagram_code": ""
   "has_sufficient_context": false
4. You MUST return a JSON object strictly conforming to this schema:
{{
    "schema_version": "2.0",
    "topic": "{question}",
    "content_type": "conceptual_definition" | "process_workflow" | "architectural_system" | "comparative_analysis",
    "domain": "Domain category (e.g. Operating Systems / Computer Networks)",
    "executive_summary": "Crisp 1-2 sentence core conceptual summary",
    "key_entities": ["Entity1", "Entity2", "Entity3"],
    "sections": [
        {{
            "heading": "Section Heading",
            "type": "key_value" | "table" | "ordered_steps" | "bullet_points",
            "items": [{{"key": "Key Name", "value": "Value description"}}],
            "table_data": {{"headers": ["Col1", "Col2"], "rows": [["Val1", "Val2"]]}},
            "steps": [{{"step_number": 1, "phase_name": "Phase Name", "action": "Action details", "resulting_state": "Result state"}}],
            "points": ["Structured point 1", "Structured point 2"],
            "summary": "Brief section overview"
        }}
    ],
    "scheme": {{
        "target_marks": {marks},
        "components": [
            {{"component": "Rubric section name", "marks": 1.0}}
        ],
        "total_allocated": {marks}
    }},
    "diagram_code": "Valid Mermaid.js code or empty string",
    "diagram_title": "Title for diagram or empty string",
    "tips_for_full_marks": [
        "Examiner scoring takeaway 1"
    ]
}}

MARKS-BASED POINTS REQUIREMENT:
- Required number of points = {marks} × 2 = {marks * 2} points  
- Generate EXACTLY {marks * 2} distinct, meaningful, non-repetitive, factually accurate numbered points
- Number every point in the answer (1. 2. 3. ...)
- Do NOT fabricate information to reach the count — if context supports fewer facts, generate fewer but accurate points
- Verify your point count equals {marks * 2} before returning the JSON
- Add a top-level field: "generated_points_count": <actual count of points you generated>

QUESTION TYPE HANDLING:
- Define: Give the definition first as Point 1, followed by relevant operational details.
- Explain: Provide distinct points with clear explanations.
- Compare: Use a comparison table or comparative points.
- List: Provide the requested items directly and concisely.
- Discuss: Include relevant arguments, explanations, and a concluding evaluation.
- Numerical problems: Show formulas, calculations, working steps, and the final answer without adding irrelevant points.

FORMAT the "answer_markdown" field as:
**Question:** {question}

**Marks:** {marks}

**Answer:**

1. [First relevant point]
2. [Second relevant point]
... [Continue until {marks * 2} points are reached]

📌 **Points Generated:** <count> / {marks * 2} required ({marks} marks × 2)

Ensure the sum of component marks exactly equals {marks}. Return ONLY the JSON object.

CONTEXT FROM ACADEMIC MATERIALS:
\"\"\"
{context_text}
\"\"\"

QUESTION: {question}
"""

        response = client.models.generate_content(
            model=self.model_name,
            contents=system_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

        try:
            data = json.loads(response.text)
            
            # Check for lack of information
            if data.get("answer_markdown") == NOT_ENOUGH_INFO_MSG:
                return {
                    "question": question,
                    "marks": marks,
                    "scheme": {
                        "target_marks": marks,
                        "components": [{"component": "Relevance Verification", "marks": marks}],
                        "total_allocated": marks
                    },
                    "answer_markdown": NOT_ENOUGH_INFO_MSG,
                    "structured_output": data,
                    "is_structured": False,
                    "diagram_code": "",
                    "diagram_needed": False,
                    "diagram_title": "",
                    "tips_for_full_marks": [],
                    "has_sufficient_context": False
                }

            diagram_code = data.get("diagram_code", "").strip()
            
            # Validate diagram syntax if code was generated
            if diagram_code:
                is_valid, err_msg = DiagramEngine.validate_mermaid_syntax(diagram_code)
                if not is_valid:
                    # Retry once to fix syntax error
                    try:
                        retry_prompt = f"The previous Mermaid diagram had syntax errors: {err_msg}.\nOriginal Code:\n{diagram_code}\nPlease fix the Mermaid syntax. Return ONLY valid Mermaid code, or an empty string if context cannot support it."
                        retry_res = client.models.generate_content(
                            model=self.model_name,
                            contents=retry_prompt,
                        )
                        cleaned_retry = DiagramEngine.sanitize_mermaid_code(retry_res.text)
                        is_valid2, _ = DiagramEngine.validate_mermaid_syntax(cleaned_retry)
                        if is_valid2:
                            data["diagram_code"] = cleaned_retry
                        else:
                            data["diagram_code"] = ""
                            data["diagram_message"] = "Diagram could not be reliably validated against PDF context."
                    except Exception:
                        data["diagram_code"] = ""
                        data["diagram_message"] = "Diagram could not be reliably validated against PDF context."
                else:
                    data["diagram_code"] = DiagramEngine.sanitize_mermaid_code(diagram_code)
            else:
                data["diagram_code"] = ""

            data["diagram_needed"] = bool(data.get("diagram_code"))
            data["marks"] = marks
            data["question"] = question

            # Validate and record points count
            target_points = marks * 2
            data["target_points"] = target_points
            
            # Calibrate scheme components
            if "scheme" in data and "components" in data["scheme"]:
                data["scheme"]["components"] = calibrate_scheme_components(data["scheme"]["components"], marks)
                data["scheme"]["total_allocated"] = marks
                data["scheme"]["target_marks"] = marks

            # Validate and compile structured output
            is_valid, val_errors = StructuredOutputPipeline.validate_structured_output(data)
            if is_valid:
                existing_ans = data.get("answer_markdown", "")
                if not existing_ans or existing_ans == NOT_ENOUGH_INFO_MSG or "Answer:" not in existing_ans:
                    data["answer_markdown"] = StructuredOutputPipeline.render_to_markdown(data, marks)
                data["answer_markdown"] = _format_answer_markdown(
                    data["answer_markdown"], question, marks,
                    int(data.get("generated_points_count", 0) or 0), target_points,
                )
                formatted_count = len(re.findall(
                    r"(?m)^\d+\. \*\*", data["answer_markdown"]
                ))
                data["generated_points_count"] = formatted_count or int(
                    data.get("generated_points_count", 0) or 0
                )
                data["is_structured"] = True
                data["has_sufficient_context"] = True
                # Keep the nested view separate: pointing structured_output
                # back at data creates a cycle that Streamlit's st.json cannot serialize.
                data["structured_output"] = {
                    key: value for key, value in data.items()
                    if key != "structured_output"
                }
            else:
                print(f"Gemini structured output validation warnings: {val_errors}")
                # Fallback to local structured pipeline
                return None

            return data
        except Exception as err:
            print(f"Error parsing Gemini response JSON: {err}")
            return None

    def _extract_grounded_points(self, question: str, marks: int, usable_sentences: List[str], context_chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Extracts up to marks * 2 distinct, meaningful, grounded points adapted to question type.
        Prioritizes definition for define questions, comparative points for compare questions,
        and ensures no duplicate, boilerplate header, or fabricated statements.
        """
        target_points = marks * 2
        q_lower = question.lower()
        q_keywords = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', q_lower) if w not in STOP_WORDS]
        broad_terms = {
            'algorithm', 'system', 'process', 'approach', 'method', 'concept',
            'structure', 'scheme', 'technique', 'types', 'type', 'functions',
            'function', 'memory', 'management',
            'hardware', 'storage', 'data', 'working', 'mechanism', 'difference',
            'differences', 'compare', 'between', 'advantages', 'disadvantages'
        }
        topic_keywords = [word for word in q_keywords if word not in broad_terms] or q_keywords

        # Identify Question Type
        if any(w in q_lower for w in ["calculate", "compute", "solve", "numerical", "find"]):
            q_type = "numerical"
        elif any(w in q_lower for w in ["define", "what is", "meaning of", "concept of"]):
            q_type = "define"
        elif any(w in q_lower for w in ["compare", "difference", "distinguish", "contrast", "vs", "versus"]):
            q_type = "compare"
        elif any(w in q_lower for w in ["list", "state", "enumerate", "name the", "types of"]):
            q_type = "list"
        elif any(w in q_lower for w in ["discuss", "evaluate", "critique"]):
            q_type = "discuss"
        else:
            q_type = "explain"

        institutional_markers = [
            "siddaganga institute of technology",
            "department of computer science",
            "course:",
            "course code",
            "autonomous institution",
            "visvesvaraya",
            "academic year",
            "dr. shivakumara",
            "affiliated to vtu"
        ]

        # 1. Collect candidate statements from context sentences and chunks
        raw_candidates = []
        
        # Helper to clean candidate
        def sanitize_candidate(text: str) -> Optional[str]:
            c = _clean_answer_text(text)
            c = re.sub(r"^[•\uf0b7\uf0a7▪◦\-*]+\s*", "", c).strip()
            # Require whitespace after a list number so section numbers such as
            # 1.2 are not partially stripped into misleading point text.
            c = re.sub(r"^\d{1,3}[.)]\s+", "", c).strip()
            c = re.sub(r"(?m)^\s*\d+\.\d+\s+[^\n]+(?:\n|$)", "", c).strip()
            # Headings and transition fragments are useful retrieval context,
            # but they are not answer points on their own.
            if re.match(r"(?i)^(?:here are|here is|for example,?\s+a\b|features (?:that are )?used to)\b", c):
                return None
            if re.match(r"(?i)^all about\b", c):
                return None
            if re.match(r"(?i)^database management system\s*\(dbms\)\s+introduction to\b", c):
                return None
            if c.rstrip().endswith(":") or c.count("(") > c.count(")"):
                return None
            words = re.findall(r"[A-Za-z][A-Za-z'-]*", c)
            title_case_ratio = (
                sum(word[0].isupper() for word in words) / len(words)
                if words else 0
            )
            if (
                len(words) <= 10
                and title_case_ratio >= 0.7
                and not re.search(r"[:.!?]", c)
            ):
                return None
            # Remove leading institutional boilerplate if any
            for im in institutional_markers:
                if im in c.lower() and len(c) < 140:
                    return None
            # Strip module headers and section titles like "MODULE 2: ... 2.1 ..."
            c = re.sub(r'(?i)^(?:module|unit|chapter)\s*\d+:[^\n]*\n?', '', c).strip()
            # Strip bullets
            if len(c) < 18:
                return None
            if any(im in c.lower() for im in ["siddaganga institute", "visvesvaraya technological", "department of computer"]):
                return None
            return c

        for s in usable_sentences:
            s_clean = s.strip()
            san = sanitize_candidate(s_clean)
            if san:
                raw_candidates.append(san)

        # 2. Question-type specific ordering
        ordered_candidates = []
        if q_type == "define":
            # Score and rank candidates by matching question keywords
            def def_score(cand: str) -> int:
                c_low = cand.lower()
                kw_score = sum(2 for k in q_keywords if re.search(r'\b' + re.escape(k) + r'\b', c_low))
                verb_score = 1 if any(w in c_low for w in ["is a", "is defined as", "refers to", "scheme that", "capability of"]) else 0
                return kw_score + verb_score

            def_candidates = [c for c in raw_candidates if def_score(c) > 0]
            def_candidates.sort(key=def_score, reverse=True)
            other_candidates = [c for c in raw_candidates if c not in def_candidates]
            ordered_candidates = def_candidates + other_candidates
        elif q_type == "compare":
            # Prioritize comparative statements
            comp_candidates = [
                c for c in raw_candidates 
                if any(w in c.lower() for w in ["whereas", "while", "unlike", "differs", "contrast", "instead of", "compared to"])
            ]
            other_candidates = [c for c in raw_candidates if c not in comp_candidates]
            ordered_candidates = comp_candidates + other_candidates
        elif q_type == "list":
            ordered_candidates = sorted(raw_candidates, key=lambda x: len(x))
        else:
            # Prefer statements that actually mention the asked subject. Retrieved
            # chunks can contain adjacent topics, so do not fill the answer with
            # unrelated fallback sentences.
            ordered_candidates = sorted(
                raw_candidates,
                key=lambda cand: sum(bool(re.search(r"\b" + re.escape(k) + r"\b", cand.lower())) for k in q_keywords),
                reverse=True,
            )

        # 4. Deduplicate and clean
        final_points = []
        seen = set()
        for c in ordered_candidates:
            clean = c.strip().rstrip(".")
            if topic_keywords and not any(
                re.search(r"\b" + re.escape(k) + r"\w*\b", clean.lower())
                or (k.endswith("ing") and re.search(r"\b" + re.escape(k[:-3] + "e") + r"\b", clean.lower()))
                for k in topic_keywords
            ):
                continue
            # When a source labels the same concept in a heading, avoid
            # repeating it just because the explanation is phrased differently.
            label = clean.split(":", 1)[0]
            norm = re.sub(r'[^a-zA-Z0-9]', '', label.lower())
            if len(clean) > 18 and norm not in seen:
                seen.add(norm)
                final_points.append(clean)
            if len(final_points) >= target_points:
                break

        return final_points

    def _generate_with_synthesis_engine(self, question: str, marks: int, context_chunks: List[Dict[str, Any]], 
                                       scheme_template: Dict, diagram_needed: bool) -> Dict[str, Any]:
        """
        Academic synthesis engine that formulates structured answers strictly from document context
        without inventing statements, generic fillers, or unsupported diagrams.
        """
        combined_text = "\n\n".join([c["text"] for c in context_chunks])
        
        # Extract keywords from question to check relevance
        GENERIC_ACADEMIC_TERMS = {
            'algorithm', 'system', 'process', 'approach', 'method', 'concept',
            'structure', 'scheme', 'technique', 'types', 'type', 'functions',
            'function', 'memory', 'management',
            'hardware', 'storage', 'data', 'working', 'mechanism'
        }
        q_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())) - STOP_WORDS
        text_lower = combined_text.lower()
        
        # Word boundary matching for precision
        matching_keywords = [k for k in q_tokens if re.search(r'\b' + re.escape(k) + r'\b', text_lower)]
        non_generic_tokens = q_tokens - GENERIC_ACADEMIC_TERMS

        # Require that specific, non-generic subject keywords are present in context
        if non_generic_tokens:
            matching_specific = [k for k in non_generic_tokens if re.search(r'\b' + re.escape(k) + r'\b', text_lower)]
            is_relevant_context = (len(matching_specific) / len(non_generic_tokens)) >= 0.50
        else:
            is_relevant_context = len(matching_keywords) > 0
        
        # Split sentences from context
        # Treat explicit PDF bullet markers as natural item boundaries. Keep
        # sentence boundaries intact and preserve newlines between chunks.
        sentences = [
            s.strip() for s in re.split(
                r'(?<=[.?!])\s+|(?=[•\uf0b7\uf0a7▪◦])|(?<=\n)-\s+|(?<=[.:]\s)-\s+',
                combined_text
            ) if len(s.strip()) > 15
        ]
        
        # Filter relevant sentences containing question keywords
        relevant_sentences = []
        for s in sentences:
            s_lower = s.lower()
            if (
                any(re.search(r"\b" + re.escape(k) + r"\b", s_lower) for k in matching_keywords)
                and not (
                    "features" in q_tokens
                    and re.search(r"\b(?:disadvantages?|complexity|overhead)\b", s_lower)
                    and not re.search(r"\b(?:disadvantages?|complexity|overhead)\b", question.lower())
                )
                and not (
                    re.search(r"\bno\s*sql\b", s_lower)
                    and not re.search(r"\bno\s*sql\b", question.lower())
                    and not re.search(r"\btypes?\b", question.lower())
                )
            ):
                relevant_sentences.append(s)

        # Never fill a marks quota from arbitrary sentences in retrieved chunks.
        # If no retrieved sentence matches the question, the answer should be
        # reported as insufficient instead of mixing in adjacent topics.
        usable_sentences = relevant_sentences

        # A modifier such as "address" can match a DBMS table column while the
        # requested subject is paging. When a subject and domain modifier both
        # occur in retrieved text, require them to co-occur in each candidate
        # sentence before calling the context sufficient.
        broad_terms = GENERIC_ACADEMIC_TERMS | {'difference', 'differences', 'compare', 'between', 'advantages', 'disadvantages'}
        topic_terms = [k for k in q_tokens if k not in broad_terms]
        domain_terms = {'memory', 'management', 'address', 'translation', 'hardware', 'storage'}
        available_domain_terms = [
            k for k in q_tokens & domain_terms
            if re.search(r'\b' + re.escape(k) + r'\b', text_lower)
        ]
        if topic_terms and available_domain_terms:
            def has_topic(sentence: str) -> bool:
                s = sentence.lower()
                return any(
                    re.search(r'\b' + re.escape(k) + r'\w*\b', s)
                    or (k.endswith('ing') and re.search(r'\b' + re.escape(k[:-3] + 'e') + r'\b', s))
                    for k in topic_terms
                )
            usable_sentences = [
                s for s in usable_sentences
                if has_topic(s) and any(re.search(r'\b' + re.escape(k) + r'\b', s.lower()) for k in available_domain_terms)
            ]

        # If insufficient topical coverage or no usable academic text, state insufficient information
        if not is_relevant_context or not usable_sentences:
            return {
                "question": question,
                "marks": marks,
                "scheme": {
                    "target_marks": marks,
                    "components": [
                        {"component": "Relevance & Content Verification", "marks": marks}
                    ],
                    "total_allocated": marks
                },
                "answer_markdown": NOT_ENOUGH_INFO_MSG,
                "structured_output": {
                    "schema_version": "2.0",
                    "topic": question,
                    "content_type": "conceptual_definition",
                    "domain": "Academic Repository",
                    "executive_summary": NOT_ENOUGH_INFO_MSG,
                    "key_entities": [],
                    "sections": [],
                    "technical_attributes": {},
                    "missing_or_uncertain_fields": ["All required content is absent from the selected document."],
                    "examiner_takeaways": ["Verify that your question is covered in the active document."]
                },
                "is_structured": False,
                "diagram_code": "",
                "diagram_needed": False,
                "diagram_title": "",
                "diagram_message": "No diagram generated because relevant content was not found in the uploaded document.",
                "tips_for_full_marks": [
                    "Ensure the question is directly covered in the uploaded course materials."
                ],
                "sources": context_chunks,
                "has_sufficient_context": False
            }

        # Attempt to synthesize grounded diagram from context
        diagram_code = ""
        diagram_title = ""
        diagram_msg = None
        if diagram_needed:
            code, title, msg = DiagramEngine.generate_diagram_from_context(question, context_chunks)
            if code:
                diagram_code = code
                diagram_title = title
            else:
                diagram_msg = msg

        # Calculate target points
        target_points = marks * 2

        # Execute Structured Output Pipeline: Content Analysis -> Schema Selection -> Transformation
        analysis = StructuredOutputPipeline.analyze_content(question, context_chunks)
        structured_data = StructuredOutputPipeline.generate_structured_data(
            question=question,
            marks=marks,
            context_chunks=context_chunks,
            analysis=analysis,
            target_points=target_points
        )

        # Validate generated structure
        is_valid, validation_errors = StructuredOutputPipeline.validate_structured_output(structured_data)
        if not is_valid:
            print(f"Structured output validation errors: {validation_errors}")

        # Build marks-based numbered answer from usable sentences respecting question type
        numbered_points = self._extract_grounded_points(question, marks, usable_sentences, context_chunks)
        generated_count = len(numbered_points)

        # Update structured data point counts
        structured_data["required_points"] = target_points
        structured_data["generated_points"] = generated_count

        # Build answer_markdown in required format
        raw_answer = "\n".join(f"{i}. {pt}." for i, pt in enumerate(numbered_points, 1))
        full_answer = _format_answer_markdown(raw_answer, question, marks, generated_count, target_points)

        # Determine official mark components
        if marks == 2:
            components = [
                {"component": "Formal Definition / Core Technical Concept", "marks": 1.0},
                {"component": "Key Characteristic / Operational Principle", "marks": 1.0}
            ]
        elif marks == 5:
            components = [
                {"component": "Introduction & Formal Definition", "marks": 1.0},
                {"component": "Working Mechanism & Core Operational Points", "marks": 2.5},
                {"component": "Technical Details & Context", "marks": 1.5}
            ]
        elif marks == 10:
            components = [
                {"component": "Conceptual Introduction & Theoretical Background", "marks": 1.5},
                {"component": "Neat Labeled Architecture / Flow Diagram", "marks": 3.0},
                {"component": "Detailed Technical Breakdown & Step-by-Step Working", "marks": 3.5},
                {"component": "Operational Rules & System Analysis", "marks": 2.0}
            ]
        else: # 15 marks
            components = [
                {"component": "In-depth Theoretical Foundation & System Principles", "marks": 2.0},
                {"component": "Exhaustive System Architecture & Labeled Diagram", "marks": 4.0},
                {"component": "Detailed Multi-Stage Component & Protocol Analysis", "marks": 5.0},
                {"component": "Operational Workflow & Technical Details", "marks": 2.5},
                {"component": "Comparative Analysis, Trade-offs & Concluding Remarks", "marks": 1.5}
            ]

        components = calibrate_scheme_components(components, marks)

        return {
            "question": question,
            "marks": marks,
            "scheme": {
                "target_marks": marks,
                "components": components,
                "total_allocated": marks
            },
            "structured_output": structured_data,
            "is_structured": True,
            "answer_markdown": full_answer,
            "diagram_code": diagram_code,
            "diagram_needed": bool(diagram_code),
            "diagram_title": diagram_title,
            "diagram_message": diagram_msg,
            "target_points": target_points,
            "generated_points_count": generated_count,
            "tips_for_full_marks": [
                f"Structure your response strictly following the {marks}-mark rubric.",
                "Highlight key terminology in bold during written examinations.",
                f"Examiners allot {components[1]['marks'] if len(components) > 1 else 1} marks specifically for neat schematics where applicable."
            ],
            "sources": context_chunks,
            "has_sufficient_context": True
        }
