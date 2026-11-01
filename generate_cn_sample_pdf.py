import pymupdf as fitz
from pathlib import Path
from config import UPLOADS_DIR

def create_cn_sample_pdf(output_path: str):
    doc = fitz.open()

    # Page 1: Module 1 - OSI Reference Model & Physical Layer
    page1 = doc.new_page()
    rect = fitz.Rect(50, 50, 550, 750)
    text1 = """SIDDAGANGA INSTITUTE OF TECHNOLOGY
DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
COURSE: COMPUTER NETWORKS (21CS52)

MODULE 1: NETWORK ARCHITECTURES AND OSI MODEL

1.1 Concept of Computer Networks
A computer network is an interconnected collection of autonomous computers capable of exchanging information and sharing computing resources.

1.2 The OSI Seven-Layer Reference Model
The Open Systems Interconnection (OSI) model developed by ISO standardizes network communication into 7 distinct layers:
1. Physical Layer: Transmits raw bitstreams over physical transmission media (cables, optical fibers, wireless).
2. Data Link Layer: Responsible for node-to-node framing, error detection (CRC), and Media Access Control (MAC).
3. Network Layer: Handles logical addressing (IPv4, IPv6) and routing packets across intermediate networks.
4. Transport Layer: Provides end-to-end process-to-process communication, connection management, flow control, and reliability (TCP, UDP).
5. Session Layer: Manages dialog control, synchronization tokens, and session checkpointing.
6. Presentation Layer: Formats, encrypts, and compresses data representations (ASCII, TLS encryption, JPEG).
7. Application Layer: Provides network services directly to user applications (HTTP, DNS, SMTP, FTP).

1.3 Transmission Media and Network Topologies
- Guided Media: Twisted pair cables, coaxial cables, and fiber optic waveguides.
- Unguided Media: Radio waves, microwaves, and infrared.
- Topologies: Mesh, Star, Bus, and Ring topologies. Star topology offers single-point fault isolation.
"""
    page1.insert_textbox(rect, text1, fontsize=10, fontname="helv")

    # Page 2: Module 2 - Data Link Layer & Sliding Window Protocols
    page2 = doc.new_page()
    text2 = """MODULE 2: DATA LINK CONTROL AND ERROR HANDLING

2.1 Error Detection and Cyclic Redundancy Check (CRC)
Cyclic Redundancy Check is a polynomial code used to detect transmission errors in digital networks.
Given a data frame D of k bits and a generator polynomial G of degree r:
1. Append r zero bits to the low-order end of frame D.
2. Divide the bit string by polynomial G using modulo-2 binary arithmetic.
3. Subtract the remainder R from the padded frame to generate the transmitted codeword T.
4. The receiver divides received codeword T by G; a non-zero remainder indicates transmission corruption.

2.2 Flow Control: Sliding Window Protocols
Flow control mechanisms prevent a fast sender from overwhelming a slow receiver.
1. Stop-and-Wait Protocol: Sender transmits 1 frame and waits for an acknowledgment (ACK) before transmitting the next. Has poor link utilization over long distance channels.
2. Go-Back-N Protocol: Sender maintains a sliding window of size N. Upon timeout of an unacknowledged frame, the sender retransmits all frames from the timed-out frame onwards. Receiver window size is 1.
3. Selective Repeat Protocol: Receiver accepts out-of-order frames within its receive window. Sender retransmits only corrupted or lost individual frames, conserving channel bandwidth.
"""
    page2.insert_textbox(rect, text2, fontsize=10, fontname="helv")

    # Page 3: Module 3 - Transport Layer & TCP Congestion Control
    page3 = doc.new_page()
    text3 = """MODULE 3: TRANSPORT LAYER PROTOCOLS (TCP & UDP)

3.1 Transmission Control Protocol (TCP)
TCP is a connection-oriented, reliable, byte-stream transport protocol.
Key features of TCP:
- Three-Way Handshake Connection Establishment: SYN, SYN-ACK, ACK.
- Connection Teardown: FIN, ACK, FIN, ACK.
- Full-duplex byte stream transmission with sequence numbers and cumulative acknowledgments.

3.2 User Datagram Protocol (UDP)
UDP is a connectionless, lightweight, unreliable transport protocol. It eliminates connection establishment latency and packet retransmission overhead. Used for real-time video streaming, VoIP, and DNS queries.

3.3 TCP Congestion Control Algorithm
Developed by Van Jacobson to prevent congestion collapse in the Internet:
1. Slow Start Phase: Congestion window (cwnd) begins at 1 Maximum Segment Size (MSS) and doubles every Round Trip Time (RTT), experiencing exponential growth until reaching slow-start threshold (ssthresh).
2. Congestion Avoidance Phase: When cwnd >= ssthresh, cwnd increases linearly by 1 MSS per RTT (Additive Increase).
3. Fast Retransmit and Fast Recovery: Reception of 3 duplicate ACKs triggers immediate retransmission of the missing segment without waiting for the retransmission timer to expire. ssthresh is set to cwnd / 2 (Multiplicative Decrease - AIMD).
"""
    page3.insert_textbox(rect, text3, fontsize=10, fontname="helv")

    doc.save(output_path)
    doc.close()
    print(f"Computer Networks sample PDF created at: {output_path}")

if __name__ == "__main__":
    out_file = UPLOADS_DIR / "Sample_Computer_Networks_Notes.pdf"
    create_cn_sample_pdf(str(out_file))
