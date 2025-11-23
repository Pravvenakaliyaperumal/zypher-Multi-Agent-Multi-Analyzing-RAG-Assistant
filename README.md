# 🤖 Zypher Multi-Agent RAG Assistant  
**Document Q&A | Domain-Specific Reasoning | Workflow Execution | Memory | Multi-Agent Orchestration**

This project demonstrates a **production-style AI agent system** built with **CoreSpeed’s Zypher agent framework**, featuring:

- Retrieval-Augmented Generation (RAG)
- Multi-agent collaboration (Coordinator + Finance Agent + Medicaid Agent)
- Workflow execution tool for real-world computations
- Memory + chat history for multi-turn conversations
- Domain-adaptive behavior (Finance / Healthcare / Medicaid)
- Strong prompt engineering + guardrails

This exceeds a standard Q&A bot and showcases how enterprise AI assistants are built in **banking, healthcare, trading, and compliance operations**.

---

## 🚀 Key Capabilities

### ✔ RAG: Retrieval-Augmented Generation  
The system loads a document, embeds it using Zypher’s `TextEmbedder`, stores it in a `VectorStore`, and retrieves relevant chunks for each user query.

### ✔ Multi-Agent Architecture  
The system uses three agents:

| Agent | Purpose |
|-------|----------|
| **CoordinatorAgent** | Classifies the query and routes it to the correct domain agent |
| **FinanceAgent** | Handles AML/KYC, trading, compliance, risk, calculations |
| **MedicaidAgent** | Handles Medicaid policies, eligibility rules, benefits, healthcare topics |

### ✔ Domain-Specific Expertise  
Each agent has its own **persona**, **tone**, and **rules**, leading to realistic and domain-appropriate answers.

### ✔ Workflow Execution Tool (Second Tool)  
A custom Zypher tool executes real workflows:

- `calculate_simple_interest(principal, rate, time)`
- `medicaid_income_check(income, threshold=18000)`

This demonstrates *agent → tool* collaboration, similar to Google ADK or CrewAI.

### ✔ Memory & Chat History  
The system uses:
- Zypher `Memory(long_term=True)`  
- Manual chat history list  
Follow-up questions work naturally.

### ✔ Guardrails  
The agent will:
- Never hallucinate  
- Refuse irrelevant requests  
- Cite document context  
- Follow internal reasoning (hidden chain of thought)

---

## 🧱 Architecture Diagram

