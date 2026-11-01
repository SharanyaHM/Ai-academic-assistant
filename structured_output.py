import re
import json
from typing import Dict, Any, List, Optional, Tuple

class StructuredOutputPipeline:
    """
    Structured Output Generation Pipeline.
    Transforms raw extracted document chunks into validated, typed,
    domain-specific structured academic data and user-friendly representations.
    
    Pipeline Stages:
    Input Chunks -> Content Analysis -> Schema Selection -> Structured Data Generation -> Validation -> UI Rendering
    """

    SUPPORTED_CONTENT_TYPES = [
        "conceptual_definition",
        "process_workflow",
        "architectural_system",
        "comparative_analysis"
    ]

    @classmethod
    def analyze_content(cls, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stage 1: Content Analysis
        Analyzes the extracted context text and question to determine:
        - Domain category
        - Purpose and content type
        - Key entities and terminology
        - Sequential steps / operational stages
        - Component relationships and properties
        """
        combined_text = " ".join([c.get("text", "") for c in context_chunks])
        text_lower = combined_text.lower()
        q_lower = question.lower()

        # 1. Determine Content Type
        if any(w in q_lower for w in ["difference", "compare", "versus", "vs", "distinguish", "contrast"]):
            content_type = "comparative_analysis"
        elif any(w in q_lower for w in ["algorithm", "lifecycle", "transition", "steps", "workflow", "banker", "handshake", "phase", "procedure", "scheduling"]):
            content_type = "process_workflow"
        elif any(w in q_lower for w in ["architecture", "hardware", "layer", "layers", "mmu", "osi", "tcp", "paging", "structure", "subsystem"]):
            content_type = "architectural_system"
        else:
            content_type = "conceptual_definition"

        # 2. Identify Academic Domain
        domain = "Computer Science & Engineering"
        if any(w in text_lower for w in ["page", "mmu", "process", "deadlock", "scheduler", "kernel", "thread", "virtual memory"]):
            domain = "Operating Systems & Systems Programming"
        elif any(w in text_lower for w in ["layer", "protocol", "packet", "subnet", "ip", "tcp", "udp", "csma", "ethernet", "router"]):
            domain = "Computer Communication Networks"
        elif any(w in text_lower for w in ["database", "sql", "transaction", "acid", "relation", "table"]):
            domain = "Database Management Systems"

        # 3. Extract Key Technical Entities (acronyms, capitalized tech terms, domain keywords)
        entities = []
        # Find acronyms (2-6 uppercase letters)
        acronyms = list(set(re.findall(r'\b[A-Z]{2,6}\b', combined_text)))
        entities.extend(acronyms[:6])

        # Find significant technical multi-word terms
        tech_terms = [
            "Logical Address", "Physical Address", "Page Table", "Frame", "Memory Management Unit",
            "Long-Term Scheduler", "Short-Term Scheduler", "Context Switching", "Deadlock Avoidance",
            "Banker's Algorithm", "Safe State", "Resource Allocation", "Application Layer",
            "Transport Layer", "Network Layer", "Data Link Layer", "Physical Layer",
            "Three-Way Handshake", "CSMA/CD", "Subnet Mask", "IP Addressing"
        ]
        for term in tech_terms:
            if term.lower() in text_lower and term not in entities:
                entities.append(term)

        # 4. Extract Relationships (e.g., A maps to B, A delivers to B, A transitions to B)
        relationships = []
        rel_patterns = [
            (r'([A-Za-z\s]{3,20})\s+(?:maps|translates)\s+(?:to|into)\s+([A-Za-z\s]{3,20})', 'maps to'),
            (r'([A-Za-z\s]{3,20})\s+(?:is divided into|consists of)\s+([A-Za-z\s]{3,30})', 'subdivides into'),
            (r'([A-Za-z\s]{3,20})\s+(?:provides|ensures)\s+([A-Za-z\s]{3,25})', 'provides'),
            (r'([A-Za-z\s]{3,20})\s+(?:dispatched to|transitions to)\s+([A-Za-z\s]{3,20})', 'transitions to')
        ]
        for pattern, rel_type in rel_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for m1, m2 in matches[:3]:
                relationships.append({
                    "source": m1.strip(),
                    "relation": rel_type,
                    "target": m2.strip()
                })

        # 5. Extract Sentences for fact grounding
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', combined_text) if len(s.strip()) > 20]

        return {
            "content_type": content_type,
            "domain": domain,
            "entities": entities[:8],
            "relationships": relationships[:4],
            "sentences": sentences,
            "combined_text": combined_text
        }

    @classmethod
    def generate_structured_data(cls, question: str, marks: int, 
                                 context_chunks: List[Dict[str, Any]],
                                 analysis: Optional[Dict[str, Any]] = None,
                                 target_points: Optional[int] = None) -> Dict[str, Any]:
        """
        Stage 2 & 3: Schema Selection and Structured Data Generation.
        Transforms extracted information into a typed, machine-readable JSON structure.
        """
        if not analysis:
            analysis = cls.analyze_content(question, context_chunks)

        if target_points is None:
            target_points = marks * 2

        content_type = analysis["content_type"]
        domain = analysis["domain"]
        entities = analysis["entities"]
        sentences = analysis["sentences"]
        relationships = analysis["relationships"]

        topic = question.strip().rstrip("?").rstrip(".")

        # Define schema-conforming sections based on content_type and marks
        sections = []
        missing_fields = []
        technical_attributes = {}

        if content_type == "conceptual_definition":
            # Section 1: Definition & Fundamental Overview (Key-Value)
            core_def = sentences[0] if sentences else f"Core principle regarding {topic}."
            sections.append({
                "heading": "1. Conceptual Foundation & Core Definition",
                "type": "key_value",
                "items": [
                    {"key": "Target Concept", "value": topic},
                    {"key": "Academic Domain", "value": domain},
                    {"key": "Formal Definition", "value": core_def},
                    {"key": "Core Purpose", "value": sentences[1] if len(sentences) > 1 else "Essential mechanism in computing architectures."}
                ],
                "summary": "Formal theoretical foundation and architectural categorization."
            })

            # Section 2: Key Operational Characteristics (Bullet Points / Labeled items)
            char_items = []
            max_bullets = max(2, target_points - 2)  # reserve 2 for def section
            for i, s in enumerate(sentences[1:min(max_bullets + 1, len(sentences))]):
                char_items.append(f"** {i+1}**: {s}")
            if not char_items:
                char_items.append("Fundamental operating principles specified in course syllabus.")

            sections.append({
                "heading": "2. Key Characteristics & Architectural Principles",
                "type": "bullet_points",
                "points": char_items,
                "summary": "Core operational properties observed during system execution."
            })

            # Section 3: Technical Attributes & Practical Scope
            tech_items = [
                {"key": "Primary Scope", "value": f"Applies to {domain} system implementations."},
                {"key": "Verification Criteria", "value": "Compliance with standard state invariants and predictable execution."}
            ]
            if entities:
                tech_items.append({"key": "Key Entities Involved", "value": ", ".join(entities[:4])})
            sections.append({
                "heading": "3. Technical Context & Scope",
                "type": "key_value",
                "items": tech_items,
                "summary": "Contextual parameters governing implementation and usage."
            })

        elif content_type == "architectural_system":
            # Section 1: System Overview (Key-Value)
            overview_text = sentences[0] if sentences else f"Architectural framework of {topic}."
            sections.append({
                "heading": "1. Architectural Overview & System Role",
                "type": "key_value",
                "items": [
                    {"key": "System Subsystem", "value": topic},
                    {"key": "Domain Category", "value": domain},
                    {"key": "Core Responsibility", "value": overview_text}
                ],
                "summary": "Theoretical overview of the subsystem components."
            })

            # Section 2: Component Breakdown (Structured Table)
            # Build structured component rows from extracted entities and context
            comp_headers = ["Subsystem Component", "Primary Responsibility", "Interface / Connection", "Operational Mechanism"]
            comp_rows = []
            
            # Map known components or extracted entities to structured rows
            if "paging" in topic.lower() or "address" in topic.lower():
                comp_rows = [
                    ["CPU / Execution Unit", "Generates logical addresses during instruction cycle", "Address Bus -> MMU", "Splits into Page Number (p) and Offset (d)"],
                    ["Page Table", "Maintains translation mappings from pages to physical frames", "MMU Internal Reference", "Indexed by p to retrieve frame number (f)"],
                    ["Physical RAM", "Stores actual process code, data, and stack frames", "Physical Address Bus", "Accessed at base frame address + offset (d)"]
                ]
            elif "layer" in topic.lower() or "osi" in topic.lower() or "network" in topic.lower():
                comp_rows = [
                    ["Application Layer", "Provides end-user services and application protocols", "Presentation / Transport", "HTTP, DNS, FTP network requests"],
                    ["Transport Layer", "Guarantees end-to-end host communication and reliability", "Network Layer", "Port addressing, segmentation, TCP/UDP"],
                    ["Network Layer", "Logical addressing, routing, and path determination", "Data Link Layer", "IP routing, packet forwarding across networks"],
                    ["Data Link / Physical", "Framing, physical transmission over media", "Hardware Media", "MAC addressing, bitstream signal encoding"]
                ]
            else:
                # Dynamic component extraction from entities
                for ent in entities[:4]:
                    matching_s = next((s for s in sentences if ent.lower() in s.lower()), f"Handles operations associated with {ent}")
                    comp_rows.append([ent, matching_s[:80] + "...", "System Interconnect", "Deterministic modular execution"])

            if not comp_rows:
                comp_rows = [["Core Module", "Primary processing logic", "System Bus", "Executes according to protocol specification"]]

            sections.append({
                "heading": "2. Structural Component Matrix",
                "type": "table",
                "table_data": {
                    "headers": comp_headers,
                    "rows": comp_rows
                },
                "summary": "Detailed matrix of active structural components, interfaces, and operational roles."
            })

            # Section 3: Operational Principles & Interconnections
            inter_points = []
            for r in relationships:
                inter_points.append(f"**{r['source']}** {r['relation']} **{r['target']}**")
            for s in sentences[1:min(4, len(sentences))]:
                inter_points.append(f"**Execution Principle**: {s}")

            sections.append({
                "heading": "3. Step-by-Step Data Flow & Component Interconnections",
                "type": "bullet_points",
                "points": inter_points[:5] if inter_points else ["Coordinates data transfer across privilege boundaries."],
                "summary": "Flow of signals, data packets, or addresses across modular boundaries."
            })

        elif content_type == "process_workflow":
            # Section 1: Workflow Foundation & Objectives
            sections.append({
                "heading": "1. Operational Process Foundation",
                "type": "key_value",
                "items": [
                    {"key": "Procedure / Algorithm", "value": topic},
                    {"key": "Classification", "value": f"{domain} Workflow"},
                    {"key": "Governing Objective", "value": sentences[0] if sentences else "Guarantees deterministic, safe state execution."}
                ],
                "summary": "Scope and theoretical basis of the workflow."
            })

            # Section 2: Step-by-Step Sequential Execution (Ordered Steps)
            step_patterns = re.findall(r'(?:step\s*(\d+)[:\-\.]?\s*([^.\n;]{10,90}))', analysis["combined_text"], re.IGNORECASE)
            steps = []
            if step_patterns:
                for num, desc in step_patterns[:5]:
                    steps.append({
                        "step_number": int(num),
                        "phase_name": f"Stage {num}",
                        "action": desc.strip(),
                        "resulting_state": "Proceeds to subsequent verification stage"
                    })
            elif "banker" in topic.lower() or "safety" in topic.lower() or "deadlock" in topic.lower():
                steps = [
                    {"step_number": 1, "phase_name": "Vector Initialization", "action": "Initialize Work = Available and Finish[i] = False for all processes i.", "resulting_state": "Baseline resources established"},
                    {"step_number": 2, "phase_name": "Condition Check", "action": "Find an index i such that Finish[i] == False and Need[i] <= Work.", "resulting_state": "Eligible process identified"},
                    {"step_number": 3, "phase_name": "Resource Reclaim", "action": "Work = Work + Allocation[i]; Finish[i] = True. Return to Step 2.", "resulting_state": "Resources returned to pool"},
                    {"step_number": 4, "phase_name": "Safety Evaluation", "action": "If Finish[i] == True for all i, system is in SAFE STATE; else UNSAFE.", "resulting_state": "Final verdict"}
                ]
            else:
                for idx, s in enumerate(sentences[1:min(5, len(sentences))], 1):
                    steps.append({
                        "step_number": idx,
                        "phase_name": f"Execution Phase {idx}",
                        "action": s,
                        "resulting_state": "Successful state progression"
                    })

            sections.append({
                "heading": "2. Step-by-Step Execution Sequence",
                "type": "ordered_steps",
                "steps": steps,
                "summary": "Formal stage-by-stage operational progression."
            })

            # Section 3: State Invariants & Boundary Rules
            sections.append({
                "heading": "3. Invariant Conditions & Correctness Guarantees",
                "type": "key_value",
                "items": [
                    {"key": "Safety Invariant", "value": "All intermediate transitions preserve system determinism."},
                    {"key": "Termination Guarantee", "value": "Finite step progression guaranteed under bounded resource conditions."},
                    {"key": "Exception Handling", "value": "Triggers safe rollback or resource wait upon condition violation."}
                ],
                "summary": "Rules governing algorithmic soundness and fault tolerance."
            })

        else: # comparative_analysis
            comp_headers = ["Comparison Dimension", "Standard Approach / System A", "Alternative Approach / System B", "Practical Impact"]
            comp_rows = [
                ["Architecture", "Monolithic / Contiguous Allocation", "Modular / Paged Allocation", "Prevents external fragmentation"],
                ["Resource Overhead", "Low translation metadata", "Page table / lookup table overhead", "Requires TLB caching to optimize"],
                ["Fault Recovery", "Manual state checkpointing", "Automated journaling / safe sequences", "Minimizes mean time to recovery"]
            ]
            sections.append({
                "heading": "1. Comparative Analysis Matrix",
                "type": "table",
                "table_data": {
                    "headers": comp_headers,
                    "rows": comp_rows
                },
                "summary": "Side-by-side evaluation across core academic parameters."
            })
            sections.append({
                "heading": "2. Trade-off Analysis & Recommendation",
                "type": "bullet_points",
                "points": [
                    "**Performance Trade-off**: Direct allocation provides slightly lower overhead, whereas structured indexing guarantees flexibility.",
                    "**Memory Utilization**: Eliminates external memory fragmentation at the expense of minor internal page fragmentation.",
                    "**Recommendation**: Follow university exam rubric guidelines by selecting the design that balances latency and resource bounds."
                ],
                "summary": "Academic takeaway for examination scoring."
            })

        # Add Examiner's Scoring Takeaways
        examiner_takeaways = [
            f"Present all {len(sections)} sections in sequence matching the university evaluation scheme.",
            "Bold technical terms and highlight equations or state names clearly in written answers.",
            f"Allocate time proportionally according to the {marks}-mark weighting."
        ]

        structured_result = {
            "schema_version": "2.0",
            "topic": topic,
            "content_type": content_type,
            "domain": domain,
            "executive_summary": sentences[0] if sentences else f"Academic overview of {topic}.",
            "key_entities": entities,
            "sections": sections,
            "technical_attributes": technical_attributes,
            "missing_or_uncertain_fields": missing_fields,
            "examiner_takeaways": examiner_takeaways
        }

        # Count total generated points across all sections
        total_generated = 0
        for sec in sections:
            st = sec.get("type")
            if st == "bullet_points":
                total_generated += len(sec.get("points", []))
            elif st == "ordered_steps":
                total_generated += len(sec.get("steps", []))
            elif st == "key_value":
                total_generated += len(sec.get("items", []))

        structured_result["required_points"] = target_points
        structured_result["generated_points"] = total_generated

        return structured_result

    @classmethod
    def validate_structured_output(cls, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Stage 4: Validation
        Ensures the generated structured data adheres to required fields,
        valid data types, and logical organization.
        """
        errors = []
        if not isinstance(data, dict):
            return False, ["Root output is not a dictionary"]

        required_keys = ["schema_version", "topic", "content_type", "domain", "sections"]
        for k in required_keys:
            if k not in data or not data[k]:
                errors.append(f"Missing or empty required top-level field: '{k}'")

        if data.get("content_type") not in cls.SUPPORTED_CONTENT_TYPES:
            errors.append(f"Invalid content_type: '{data.get('content_type')}'. Supported: {cls.SUPPORTED_CONTENT_TYPES}")

        sections = data.get("sections", [])
        if not isinstance(sections, list) or len(sections) == 0:
            errors.append("Sections field must be a non-empty list of structured section objects.")
        else:
            for idx, sec in enumerate(sections):
                if not isinstance(sec, dict):
                    errors.append(f"Section {idx} is not a valid dictionary.")
                    continue
                if "heading" not in sec:
                    errors.append(f"Section {idx} is missing 'heading'.")
                if "type" not in sec:
                    errors.append(f"Section {idx} is missing 'type'.")

                stype = sec.get("type")
                if stype == "table":
                    tdata = sec.get("table_data", {})
                    if not isinstance(tdata, dict) or "headers" not in tdata or "rows" not in tdata:
                        errors.append(f"Section {idx} of type 'table' is missing valid table_data.")
                elif stype == "ordered_steps":
                    if not isinstance(sec.get("steps"), list):
                        errors.append(f"Section {idx} of type 'ordered_steps' is missing steps list.")
                elif stype == "key_value":
                    if not isinstance(sec.get("items"), list):
                        errors.append(f"Section {idx} of type 'key_value' is missing items list.")
                elif stype == "bullet_points":
                    if not isinstance(sec.get("points"), list):
                        errors.append(f"Section {idx} of type 'bullet_points' is missing points list.")

        # Warn (not fail) if required_points vs generated_points differ significantly
        warnings = []
        required = data.get("required_points")
        generated = data.get("generated_points")
        if required and generated and abs(required - generated) > required * 0.5:
            warnings.append(f"WARNING: Generated {generated} points but required {required} (marks × 2). Consider reviewing context coverage.")

        return len(errors) == 0, errors + warnings

    @classmethod
    def render_to_markdown(cls, structured_data: Dict[str, Any], marks: int) -> str:
        """
        Converts the validated structured data into cleanly formatted, mark-calibrated
        Markdown adhering strictly to the university exam structure.
        """
        target_points = structured_data.get("required_points", marks * 2)
        generated_points = structured_data.get("generated_points", target_points)
        topic = structured_data.get('topic', 'Academic Analysis')

        parts = []

        # Standard Exam Question & Marks Header
        parts.append(f"**Question:** {topic}\n")
        parts.append(f"**Marks:** {marks}\n")
        parts.append("**Answer:**\n")

        # Academic Overview
        if structured_data.get('executive_summary'):
            parts.append(f"> **Overview**: {structured_data.get('executive_summary')}\n")

        # Key entities chips
        if structured_data.get("key_entities"):
            chips = " • ".join([f"`{e}`" for e in structured_data["key_entities"][:6]])
            parts.append(f"🔑 **Key Entities & Symbols**: {chips}\n")

        # Sections
        point_idx = 1
        for sec in structured_data.get("sections", []):
            parts.append(f"### {sec.get('heading', 'Section')}")
            if sec.get("summary"):
                parts.append(f"*{sec['summary']}*\n")

            stype = sec.get("type")
            if stype == "key_value":
                for item in sec.get("items", []):
                    parts.append(f"{point_idx}. **{item['key']}**: {item['value']}")
                    point_idx += 1
                parts.append("")

            elif stype == "table":
                tdata = sec.get("table_data", {})
                headers = tdata.get("headers", [])
                rows = tdata.get("rows", [])
                if headers and rows:
                    parts.append("| " + " | ".join(headers) + " |")
                    parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for r in rows:
                        parts.append("| " + " | ".join([str(cell).replace("|", "/") for cell in r]) + " |")
                    parts.append("")

            elif stype == "ordered_steps":
                for s in sec.get("steps", []):
                    parts.append(f"**Step {s.get('step_number', point_idx)}: {s.get('phase_name', 'Phase')}**")
                    parts.append(f"- *Action*: {s.get('action', '')}")
                    parts.append(f"- *Resulting State*: `{s.get('resulting_state', '')}`\n")
                    point_idx += 1

            elif stype == "bullet_points":
                for pt in sec.get("points", []):
                    parts.append(f"{point_idx}. {pt}")
                    point_idx += 1
                parts.append("")

        parts.append(f"📌 **Points Generated:** {generated_points} / {target_points} required ({marks} marks × 2)")
        return "\n".join(parts)
