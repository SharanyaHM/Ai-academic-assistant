import re
from typing import Dict, Any, Optional, Tuple, List

class DiagramEngine:
    """
    Academic Diagram Engine.
    Generates verified Mermaid.js diagrams strictly grounded in retrieved PDF context.
    Eliminates generic/unrelated fallback diagrams and validates syntax before rendering.
    """

    DIAGRAM_KEYWORDS = [
        "diagram", "architecture", "flowchart", "workflow", "lifecycle",
        "pipeline", "layers", "stack", "model", "schema", "protocol",
        "state", "hierarchy", "network", "paging", "segmentation",
        "deadlock", "tcp/ip", "osi", "hardware"
    ]

    @classmethod
    def should_include_diagram(cls, question: str, marks: int, 
                               context_chunks: Optional[List[Dict[str, Any]]] = None,
                               forced: bool = False) -> bool:
        """Determines if a diagram is appropriate and supported by available context."""
        if context_chunks is not None and not context_chunks:
            return False
            
        if forced:
            return True
        if marks >= 10:
            return True
        q_lower = question.lower()
        if any(kw in q_lower for kw in cls.DIAGRAM_KEYWORDS):
            return True
        if marks == 5 and any(w in q_lower for w in ["explain", "working", "how", "steps", "types"]):
            return True
        return False

    @classmethod
    def sanitize_mermaid_code(cls, code: str) -> str:
        """Cleans and normalizes Mermaid.js code strings."""
        if not code:
            return ""

        # Remove markdown fences if model returned them
        code = re.sub(r"^```(?:mermaid)?", "", code, flags=re.MULTILINE)
        code = re.sub(r"```$", "", code, flags=re.MULTILINE).strip()

        # Fix unescaped parentheses inside node labels e.g., A[Process (P1)] -> A["Process (P1)"]
        code = re.sub(r'\[([^"\]\n]*\([^\)\n]*\)[^"\]\n]*)\]', r'["\1"]', code)

        return code

    @classmethod
    def validate_mermaid_syntax(cls, code: str) -> Tuple[bool, str]:
        """
        Validates Mermaid diagram syntax.
        Checks for valid header, delimiter matching, and structural completeness.
        Returns (is_valid, error_message).
        """
        if not code or not code.strip():
            return False, "Diagram code is empty."

        clean = cls.sanitize_mermaid_code(code).strip()
        lines = [line.strip() for line in clean.split("\n") if line.strip() and not line.strip().startswith("%%")]
        if not lines:
            return False, "Diagram contains no executable Mermaid statements."

        valid_headers = (
            "graph", "flowchart", "sequenceDiagram", "classDiagram",
            "stateDiagram-v2", "stateDiagram", "erDiagram", "gantt", "pie"
        )
        header_line = lines[0].strip()
        if not any(header_line.startswith(h) for h in valid_headers):
            return False, f"Invalid Mermaid header '{header_line}'. Must start with one of: {', '.join(valid_headers)}."

        # Verify balanced quotes and brackets across lines
        for idx, line in enumerate(lines, 1):
            # Check quotes
            quotes = len(re.findall(r'(?<!\\)"', line))
            if quotes % 2 != 0:
                return False, f"Line {idx} has unclosed quotes: {line}"

            # Check brackets outside quotes
            unquoted = re.sub(r'"[^"]*"', '', line)
            for open_b, close_b in [('[', ']'), ('(', ')'), ('{', '}')]:
                if unquoted.count(open_b) != unquoted.count(close_b):
                    return False, f"Line {idx} has mismatched bracket '{open_b}{close_b}': {line}"

        # Ensure multi-line structural content (not just a single header)
        if len(lines) < 2 and not any(h in header_line for h in ["pie", "gantt"]):
            return False, "Diagram contains only a header line with no components or connections."

        return True, ""

    @classmethod
    def generate_diagram_from_context(cls, question: str, 
                                      context_chunks: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Synthesizes a relevant Mermaid diagram strictly from the question and retrieved PDF context.
        Returns:
            (mermaid_code, diagram_title, info_or_error_message)
        If the PDF lacks sufficient architectural/relational details, returns (None, None, message).
        """
        if not context_chunks:
            return None, None, "The uploaded PDF does not contain enough architectural or workflow details to construct a reliable diagram for this question."

        combined_text = " ".join([c.get("text", "") for c in context_chunks])
        text_lower = combined_text.lower()
        q_lower = question.lower()

        # 1. Paging & Address Translation (Grounded in PDF text)
        is_paging_q = any(k in q_lower for k in ["paging", "translation", "page table", "mmu", "address"])
        has_paging_evidence = (
            ("page" in text_lower or "frame" in text_lower) and
            ("logical address" in text_lower or "physical address" in text_lower or "page table" in text_lower)
        )
        if is_paging_q and has_paging_evidence:
            code = """flowchart LR
    subgraph CPU ["CPU Generated Address"]
        VA["Logical Address<br/>[Page # (p) | Offset (d)]"]
    end
    subgraph MMU ["Memory Management Unit"]
        PT["Page Table<br/>(Maps Page p &rarr; Frame f)"]
    end
    subgraph RAM ["Physical Memory (RAM)"]
        PA["Physical Address<br/>[Frame # (f) | Offset (d)]"]
        MEM["Physical Frame"]
    end
    VA -->|Page Number p| PT
    PT -->|Frame Number f| PA
    VA -.->|Direct Offset d| PA
    PA --> MEM
    style CPU fill:#dbeafe,stroke:#2563eb,stroke-width:2px
    style MMU fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style RAM fill:#dcfce7,stroke:#16a34a,stroke-width:2px"""
            is_valid, _ = cls.validate_mermaid_syntax(code)
            if is_valid:
                return code, "Hardware Address Translation Architecture (Paging)", None

        # 2. Layered Protocol Architecture (Grounded in PDF text)
        is_network_q = any(k in q_lower for k in ["layer", "osi", "tcp", "protocol", "architecture", "network"])
        has_network_evidence = (
            ("application" in text_lower or "transport" in text_lower or "network" in text_lower) and
            ("layer" in text_lower or "protocol" in text_lower or "packet" in text_lower)
        )
        if is_network_q and has_network_evidence:
            # Build layers based on what the document actually covers
            layers = []
            if "application" in text_lower:
                layers.append(('L_APP', 'Application Layer (User Protocols, HTTP/DNS)'))
            if "transport" in text_lower:
                layers.append(('L_TRANS', 'Transport Layer (TCP/UDP End-to-End Delivery)'))
            if "network" in text_lower or "internet" in text_lower:
                layers.append(('L_NET', 'Network Layer (IP Logical Addressing & Routing)'))
            if "data link" in text_lower or "framing" in text_lower or "mac" in text_lower:
                layers.append(('L_LINK', 'Data Link Layer (Framing, Flow & Error Control)'))
            if "physical" in text_lower or "bit" in text_lower:
                layers.append(('L_PHY', 'Physical Layer (Signal & Bit Transmission)'))

            if len(layers) >= 3:
                lines = ["flowchart TD", "    subgraph Architecture [\"Layered Protocol Architecture\"]"]
                for i, (node_id, label) in enumerate(layers):
                    lines.append(f'        {node_id}["{label}"]')
                for i in range(len(layers) - 1):
                    lines.append(f'        {layers[i][0]} --> {layers[i+1][0]}')
                lines.append("    end")
                code = "\n".join(lines)
                is_valid, _ = cls.validate_mermaid_syntax(code)
                if is_valid:
                    return code, "Layered Protocol Stack & Data Flow", None

        # 3. Process State Transition Model (Grounded in PDF text)
        is_process_q = any(k in q_lower for k in ["process", "state", "lifecycle", "transition", "pcb"])
        has_process_evidence = (
            "ready" in text_lower and "running" in text_lower and
            any(s in text_lower for s in ["waiting", "terminated", "blocked", "new", "exit"])
        )
        if is_process_q and has_process_evidence:
            code = """stateDiagram-v2
    [*] --> New: Process Created
    New --> Ready: Admitted to Ready Queue
    Ready --> Running: Dispatched by CPU Scheduler
    Running --> Ready: Interrupt / Time Slice Expired
    Running --> Waiting: I/O or Event Wait
    Waiting --> Ready: I/O or Event Completed
    Running --> Terminated: Exit / Completion
    Terminated --> [*]"""
            is_valid, _ = cls.validate_mermaid_syntax(code)
            if is_valid:
                return code, "Process State Transition Lifecycle", None

        # 4. Deadlock / Banker's Algorithm Flow (Grounded in PDF text)
        is_banker_q = any(k in q_lower for k in ["banker", "deadlock", "safety", "avoidance", "resource allocation"])
        has_banker_evidence = (
            ("allocation" in text_lower or "available" in text_lower or "need" in text_lower) and
            ("safety" in text_lower or "deadlock" in text_lower or "work" in text_lower)
        )
        if is_banker_q and has_banker_evidence:
            code = """flowchart TD
    subgraph SafetyAlgorithm ["Banker's Safety & State Check Algorithm"]
        Init["Initialize: Work = Available, Finish[i] = False"] --> Find["Find Process i such that Finish[i] == False and Need[i] &le; Work"]
        Find -->|Found| Allocate["Work = Work + Allocation[i]<br/>Finish[i] = True"]
        Allocate --> Find
        Find -->|No such Process exists| CheckAll{"Finish[i] == True for all i?"}
        CheckAll -->|Yes| Safe["System is in SAFE STATE<br/>No Deadlock Occurs"]
        CheckAll -->|No| Unsafe["System is in UNSAFE STATE<br/>Potential Deadlock Detected"]
    end
    style Safe fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style Unsafe fill:#fee2e2,stroke:#ef4444,stroke-width:2px"""
            is_valid, _ = cls.validate_mermaid_syntax(code)
            if is_valid:
                return code, "Banker's Algorithm State & Safety Flowchart", None

        # 5. Dynamic extraction of sequential steps or algorithm stages from text
        # Look for explicit step sequences in the retrieved PDF context
        step_patterns = re.findall(r'(?:step\s*(\d+)[:\-\.]?\s*([^.\n;]{10,90}))', combined_text, re.IGNORECASE)
        if len(step_patterns) >= 3:
            lines = ["flowchart TD", f'    subgraph Workflow ["{question[:40].strip()}"]']
            node_ids = []
            for num, desc in step_patterns[:5]:
                clean_desc = desc.strip().replace('"', "'")
                nid = f"S{num}"
                lines.append(f'        {nid}["Step {num}: {clean_desc}"]')
                node_ids.append(nid)
            for i in range(len(node_ids) - 1):
                lines.append(f'        {node_ids[i]} --> {node_ids[i+1]}')
            lines.append("    end")
            code = "\n".join(lines)
            is_valid, _ = cls.validate_mermaid_syntax(code)
            if is_valid:
                return code, f"Operational Workflow: {question[:35]}", None

        # 6. Fallback: If context does not support a reliable diagram, DO NOT invent one
        return None, None, "The uploaded PDF does not contain enough architectural or workflow details to construct a reliable diagram for this question."

    @classmethod
    def generate_html_embed(cls, mermaid_code: str, title: str = "Academic Concept Diagram") -> str:
        """Generates an embeddable HTML snippet containing Mermaid.js with clean styling."""
        if not mermaid_code or not mermaid_code.strip():
            return ""

        clean_code = cls.sanitize_mermaid_code(mermaid_code)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    background: transparent;
                    margin: 0;
                    padding: 12px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .diagram-card {{
                    background: #ffffff;
                    border: 1px solid #e5e7eb;
                    border-radius: 6px;
                    padding: 16px;
                    width: 98%;
                }}
                .diagram-title {{
                    font-size: 13px;
                    font-weight: 600;
                    color: #374151;
                    margin-bottom: 12px;
                    border-bottom: 2px solid #2563eb;
                    padding-bottom: 4px;
                    display: inline-block;
                }}
                .mermaid {{
                    display: flex;
                    justify-content: center;
                    overflow-x: auto;
                }}
            </style>
        </head>
        <body>
            <div class="diagram-card">
                <div class="diagram-title">{title}</div>
                <div class="mermaid">
{clean_code}
                </div>
            </div>
            <script>
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'default',
                    themeVariables: {{
                        primaryColor: '#e0f2fe',
                        primaryTextColor: '#0369a1',
                        primaryBorderColor: '#0284c7',
                        lineColor: '#64748b',
                        secondaryColor: '#f1f5f9',
                        tertiaryColor: '#ffffff'
                    }}
                }});
            </script>
        </body>
        </html>
        """
        return html_content
