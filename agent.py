
# agent.py

import os
from zypher import Agent, Memory
from zypher.tools import TextEmbedder, VectorStore, Document

# Import Workflow Tool
from tools.workflow_executor import WorkflowExecutor


# ----------------------------------------------------------------------
# 🔹 1. Load All Documents Dynamically
# ----------------------------------------------------------------------

DOCUMENTS_FOLDER = "./documents/"
loaded_docs = []   # list of (filename, text)

for file_name in os.listdir(DOCUMENTS_FOLDER):
    if file_name.endswith(".txt"):
        file_path = os.path.join(DOCUMENTS_FOLDER, file_name)
        with open(file_path, "r") as f:
            text = f.read()
        loaded_docs.append((file_name, text))

if not loaded_docs:
    raise Exception("No documents found inside /documents folder!")


# ----------------------------------------------------------------------
# 🔹 2. Embed All Documents and Store in VectorStore
# ----------------------------------------------------------------------

embedder = TextEmbedder()
vectorstore = VectorStore()

for idx, (file_name, text) in enumerate(loaded_docs):
    document = Document(id=f"doc_{idx}", text=text)
    embedding = embedder.embed(text)
    vectorstore.add(
        document.id,
        embedding,
        metadata={"text": text, "source": file_name}
    )


# ----------------------------------------------------------------------
# 🔹 3. Create Workflow Tool
# ----------------------------------------------------------------------

workflow_tool = WorkflowExecutor()


# ----------------------------------------------------------------------
# 🔹 4. Define Multi-Agent System (Coordinator + Two Domain Agents)
# ----------------------------------------------------------------------

# ---------------- Finance Agent ----------------
finance_agent = Agent(
    name="FinanceAgent",
    role="""
    You are a Finance Policy Assistant specializing in:
    - AML / KYC rules
    - Trading surveillance patterns
    - Risk controls
    - Margin rules
    - SAR reporting
    - Finance calculations

    Tone: professional, compliance-oriented.

    Rules:
    - Use ONLY the retrieved document chunk + chat history.
    - Never hallucinate or invent financial rules.
    - If a calculation is required, call the workflow_executor tool.
    """,
    memory=Memory(long_term=False)
)
finance_agent.add_tool(workflow_tool)


# ---------------- Medicaid Agent ----------------
medicaid_agent = Agent(
    name="MedicaidAgent",
    role="""
    You are a Healthcare/Medicaid Policy Assistant specializing in:
    - Medicaid eligibility
    - Income thresholds
    - Covered benefits
    - Prior authorization rules

    Tone: supportive, patient-friendly.

    Rules:
    - Use ONLY the retrieved document chunk + chat history.
    - Never hallucinate medical or Medicaid facts.
    - For eligibility checks, use the workflow_executor tool.
    """,
    memory=Memory(long_term=False)
)
medicaid_agent.add_tool(workflow_tool)


# ---------------- Coordinator Agent ----------------
coordinator_agent = Agent(
    name="CoordinatorAgent",
    role="""
    You are a smart routing agent.

    Your responsibilities:
    1. Detect if question belongs to FINANCE or MEDICAID domain.
    2. Choose the correct agent:
       - FinanceAgent
       - MedicaidAgent
    3. Only when unclear, respond:
       "This question does not match the document domain."

    Domain signals:
    - Finance → AML, KYC, compliance, trading, risk, margin, wire, SAR.
    - Medicaid → eligibility, coverage, benefits, income, prior authorization.

    Always return the agent name: "finance" or "medicaid" or "unknown".
    """,
    memory=Memory(long_term=True)
)
coordinator_agent.add_tool(workflow_tool)


# ----------------------------------------------------------------------
# 🔹 5. Keyword-based Router (Fast + Simple)
# ----------------------------------------------------------------------

def determine_domain(question: str):
    q = question.lower()

    finance_words = ["aml", "kyc", "trade", "trading", "risk", "compliance", "sar", "margin"]
    medicaid_words = ["medicaid", "coverage", "benefit", "income", "eligibility", "prior authorization"]

    if any(w in q for w in finance_words):
        return "finance"

    if any(w in q for w in medicaid_words):
        return "medicaid"

    return "unknown"


# ----------------------------------------------------------------------
# 🔹 6. Chat History (Manual)
# ----------------------------------------------------------------------

chat_history = []


# ----------------------------------------------------------------------
# 🔹 7. Main Retrieval + Agent Routing Logic
# ----------------------------------------------------------------------

def answer_query(query: str):
    # Save user message
    chat_history.append({"user": query})

    # Step 1: Embed question and retrieve relevant doc chunk
    q_emb = embedder.embed(query)
    retrieved = vectorstore.search(q_emb, top_k=1)

    if not retrieved:
        return "I couldn't retrieve relevant content from the document."

    top_match = retrieved[0]["metadata"]["text"]
    source_file = retrieved[0]["metadata"]["source"]

    # Step 2: Build formatted chat history
    history_text = ""
    for msg in chat_history:
        if "assistant" in msg:
            history_text += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        else:
            history_text += f"User: {msg['user']}\n"

    # Step 3: Determine domain (fast path)
    domain = determine_domain(query)

    # Step 4: Build the shared RAG prompt
    base_prompt = f"""
### Retrieved Document Chunk (from {source_file}):
{top_match}

### Conversation History:
{history_text}

### User Question:
{query}

### Instructions:
- Use ONLY the retrieved chunk + history.
- Do NOT hallucinate.
- If a workflow is required, call workflow_executor.
"""

    # Step 5: Route to correct agent
    if domain == "finance":
        response = finance_agent.run(base_prompt)

    elif domain == "medicaid":
        response = medicaid_agent.run(base_prompt)

    else:
        # Let Coordinator try to classify
        routing = coordinator_agent.run(f"Question: {query}\nReturn domain only.")
        routing = routing.lower().strip()

        if "finance" in routing:
            response = finance_agent.run(base_prompt)
        elif "medicaid" in routing:
            response = medicaid_agent.run(base_prompt)
        else:
            response = (
                "This question does not match the domain of the uploaded documents."
            )

    # Save assistant response
    chat_history.append({"assistant": response})

    return response


# ----------------------------------------------------------------------
# 🔹 8. CLI Loop
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("🤖 Multi-Agent RAG Assistant Ready!")
    print("Ask your question (type 'exit' to quit):\n")

    while True:
        user_q = input("> ")

        if user_q.lower() == "exit":
            break

        ans = answer_query(user_q)
        print("\nAssistant:", ans, "\n")
