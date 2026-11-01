import asyncio
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import settings
from langchain_agent.rag import retrieve, build_context


# ---------------------------------------------------------------------------
# Shared singletons — initialized once at import time
# ---------------------------------------------------------------------------

class _Lazy:
    """Create the object on first use, inside the process that needs it.

    Qdrant local mode only allows ONE client per storage folder. LiveKit imports this
    module in the main process AND in every job process, so opening the client at import
    time makes them fight over the lock. Opening it lazily means only the process that
    actually serves a call touches the database.
    """

    def __init__(self, factory):
        self._factory = factory
        self._obj = None

    def __getattr__(self, name):
        if self._obj is None:
            self._obj = self._factory()
        return getattr(self._obj, name)


embedder = _Lazy(lambda: SentenceTransformer(settings.embedding_model))
qdrant = _Lazy(lambda: QdrantClient(path=settings.qdrant_path))


# ---------------------------------------------------------------------------
# Mock data
# (keys are unchanged so other modules keep working; the meaning of the values
#  is mapped to health insurance:
#   data_*  -> claims used vs. sum insured (in lakh ₹)
#   voice_* -> health check-ups used vs. allowed
#   sms_*   -> wellness reward points redeemed vs. available)
# ---------------------------------------------------------------------------

MOCK_CUSTOMER_DB = {
    "9876543210": {
        "name": "Rajesh Kumar",
        "plan": "Evernorth Select 5L",
        "balance_inr": 1249.92,
        "balance_type": "postpaid",
        "next_bill_date": "2026-12-11",
        "data_usage_gb": 1.2,
        "data_limit_gb": 5.0,
        "voice_usage_min": 0,
        "voice_limit_min": 1,
        "sms_usage": 45,
        "sms_limit": 100,
    },
    "9123456789": {
        "name": "Priya Singh",
        "plan": "Evernorth Essential 3L",
        "balance_inr": 199.99,
        "balance_type": "prepaid",
        "validity_end": "2027-03-14",
        "data_usage_gb": 0.5,
        "data_limit_gb": 3.0,
        "voice_usage_min": 0,
        "voice_limit_min": 0,
        "sms_usage": 20,
        "sms_limit": 100,
    },
    "8765432109": {
        "name": "Amit Patel",
        "plan": "Evernorth Premium 10L",
        "balance_inr": 2083.25,
        "balance_type": "postpaid",
        "next_bill_date": "2026-12-15",
        "data_usage_gb": 3.8,
        "data_limit_gb": 10.0,
        "voice_usage_min": 1,
        "voice_limit_min": 2,
        "sms_usage": 60,
        "sms_limit": 100,
    },
}

MOCK_PLANS_DB = {
    "Evernorth Essential 3L": {
        "price": 8499,
        "validity": "1 year",
        "data": "₹3,00,000",
        "voice": "up to ₹3,000 per day (shared room)",
        "sms": "30 days before and 60 days after hospitalisation",
        "benefits": [
            "Cashless treatment at network hospitals",
            "Day-care procedures covered",
            "Free ambulance up to ₹1,500 per event",
        ],
    },
    "Evernorth Select 5L": {
        "price": 14999,
        "validity": "1 year",
        "data": "₹5,00,000",
        "voice": "up to ₹5,000 per day (single private room)",
        "sms": "45 days before and 90 days after hospitalisation",
        "benefits": [
            "1 full body health check-up per year",
            "No-claim bonus up to 50%",
            "10% off Evernorth Home Delivery pharmacy",
        ],
    },
    "Evernorth Premium 10L": {
        "price": 24999,
        "validity": "1 year",
        "data": "₹10,00,000",
        "voice": "no cap (single private room)",
        "sms": "60 days before and 120 days after hospitalisation",
        "benefits": [
            "Maternity cover of ₹50,000 after 24-month waiting period",
            "OPD benefit of ₹10,000 per year",
            "2 health check-ups per year and priority claims desk",
        ],
    },
    "Evernorth Elite 25L": {
        "price": 42999,
        "validity": "1 year",
        "data": "₹25,00,000 with unlimited automatic restore",
        "voice": "any room category",
        "sms": "90 days before and 180 days after hospitalisation",
        "benefits": [
            "Maternity cover of ₹1,00,000 and OPD benefit of ₹25,000",
            "Emergency hospitalisation abroad and air ambulance",
            "Dedicated relationship manager and 24/7 priority claims line",
        ],
    },
    "Evernorth Senior Care 10L": {
        "price": 31999,
        "validity": "1 year",
        "data": "₹10,00,000",
        "voice": "up to ₹7,000 per day",
        "sms": "60 days before and 90 days after hospitalisation",
        "benefits": [
            "For ages 60 to 80 with 12-month pre-existing condition waiting period",
            "Home nursing and 4 doctor-at-home visits per year",
            "Annual check-up with cardiac and bone density screening",
        ],
    },
}


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------


@tool
def search_plans(query: str) -> str:
    """Search Evernorth Health Services' plan database and documentation for information about
    health insurance plans, premiums, sum insured, waiting periods, claims, billing, and benefits.
    Use this whenever the user asks about plans, premiums, or plan comparison."""
    chunks = retrieve(qdrant, embedder, query, top_k=5)
    if not chunks:
        return "No relevant information found in the documentation."
    return build_context(chunks)


@tool
def check_account_balance(phone_number: str) -> str:
    """Check the policy status, premium due, claims used, health check-ups, and wellness points for a customer.
    Requires the 10-digit registered phone number (e.g., 9876543210)."""

    phone_number = phone_number.strip()

    if not phone_number.isdigit() or len(phone_number) != 10:
        return (
            "Invalid phone number. Please provide a valid 10-digit Indian mobile number "
            "(e.g., 9876543210)."
        )

    customer = MOCK_CUSTOMER_DB.get(phone_number)
    if not customer:
        return f"No policy found for {phone_number}. Please verify the number."

    if customer["balance_type"] == "postpaid":
        balance_info = f"Premium instalment pending. Amount due: ₹{customer['balance_inr']:.2f}"
        period = f"Next premium due date: {customer['next_bill_date']}"
    else:
        balance_info = f"Premium paid in advance. Wellness wallet balance: ₹{customer['balance_inr']:.2f}"
        period = f"Policy valid until: {customer['validity_end']}"

    usage = (
        f"Current plan: {customer['plan']}. "
        f"Claims: ₹{customer['data_usage_gb']} lakh of ₹{customer['data_limit_gb']} lakh sum insured used. "
        f"Health check-ups: {customer['voice_usage_min']} of {customer['voice_limit_min']} used this year. "
        f"Wellness reward points: {customer['sms_usage']} of {customer['sms_limit']} redeemed."
    )

    return f"{balance_info}. {period}. {usage}"


@tool
def get_plan_recommendation(usage_type: str, data_gb: int = 0, voice_min: int = 0) -> str:
    """Recommend an Evernorth Health Services plan based on the customer's health cover needs.
    Usage types: 'light' (young and healthy, first policy), 'medium' (regular check-ups),
    'heavy' (family, maternity or comprehensive cover), or specify the exact sum insured
    needed in lakh rupees using data_gb (e.g., 10 for ₹10 lakh)."""

    usage_type = usage_type.lower().strip()

    if usage_type == "light":
        recommended = "Evernorth Essential 3L"
        reason = "Perfect for a first health policy with solid base cover"
    elif usage_type == "medium":
        recommended = "Evernorth Select 5L"
        reason = "Ideal for working professionals who want regular check-ups and a no-claim bonus"
    elif usage_type == "heavy":
        recommended = "Evernorth Premium 10L or Elite 25L"
        reason = "Great for families, maternity planning, and comprehensive protection"
    else:
        if data_gb <= 3:
            recommended = "Evernorth Essential 3L"
        elif data_gb <= 5:
            recommended = "Evernorth Select 5L"
        elif data_gb <= 10:
            recommended = "Evernorth Premium 10L"
        else:
            recommended = "Evernorth Elite 25L"
        reason = f"Based on your requirement of ₹{data_gb} lakh sum insured"

    plan = MOCK_PLANS_DB.get(recommended)
    if plan:
        return (
            f"{reason}. Plan: {recommended}. "
            f"Premium: ₹{plan['price']}/year, sum insured {plan['data']}, "
            f"room rent {plan['voice']}, hospitalisation cover {plan['sms']}. "
            f"Benefits: {', '.join(plan['benefits'])}."
        )

    return f"Recommended plan: {recommended}. Please ask for more details."


@tool
def get_plan_details(plan_name: str) -> str:
    """Get detailed information about a specific Evernorth Health Services plan.
    Provide the plan name (e.g., 'Evernorth Select 5L')."""

    plan_name = plan_name.strip()

    # Try exact match first, then case-insensitive
    plan = MOCK_PLANS_DB.get(plan_name)
    if not plan:
        for key in MOCK_PLANS_DB:
            if key.lower() == plan_name.lower():
                plan = MOCK_PLANS_DB[key]
                plan_name = key
                break

    if not plan:
        return (
            f"Plan '{plan_name}' not found. Available plans: "
            f"{', '.join(MOCK_PLANS_DB.keys())}."
        )

    benefits = ", ".join(plan["benefits"])
    return (
        f"{plan_name}: ₹{plan['price']} per {plan['validity']}. "
        f"Includes sum insured of {plan['data']}, room rent {plan['voice']}, "
        f"hospitalisation cover {plan['sms']}. "
        f"Benefits: {benefits}."
    )


@tool
def escalate_to_support(reason: str) -> str:
    """Escalate to a human support representative for issues the agent cannot resolve —
    claim disputes, rejected claims, cashless approval problems, complaints, or special requests."""

    ticket_id = "EVERNORTH-" + str(abs(hash(reason)) % 100000).zfill(5)
    timestamp = datetime.now().isoformat()

    print(f"[ESCALATION] ID={ticket_id} time={timestamp} reason={reason!r}")

    return (
        f"I've created support ticket {ticket_id}. "
        f"A representative will contact you within 2 hours. "
        f"Your reference is {ticket_id}."
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a friendly and professional voice-based customer support agent for Evernorth Health Services.\n"
    "\n"
    "GUIDELINES:\n"
    "- Use search_plans to answer questions about health plans, premiums, waiting periods, claims, billing, or comparisons.\n"
    "- Use check_account_balance when the customer provides their 10-digit registered phone number.\n"
    "- Use get_plan_recommendation when the customer describes their health cover needs.\n"
    "- Use get_plan_details for detailed plan information.\n"
    "- Use escalate_to_support for issues you cannot resolve.\n"
    "\n"
    "TONE:\n"
    "- Be warm, empathetic, professional, and helpful.\n"
    "- Keep responses SHORT — this is a voice call, not text chat.\n"
    "- Use Indian currency (₹).\n"
    "\n"
    "CONSTRAINTS:\n"
    "- NEVER make up information about plans, premiums, benefits, or coverage.\n"
    "- NEVER give medical advice or diagnoses; refer medical questions to a doctor.\n"
    "- NEVER request sensitive details beyond phone number.\n"
    "- ALWAYS verify phone numbers are 10 digits before lookup.\n"
    "- If you can't find information, offer to escalate.\n"
    "- Always keep your responses conversational, don't respond with tables or diagrams."
)


# ---------------------------------------------------------------------------
# Build the LangChain agent graph
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    search_plans,
    check_account_balance,
    get_plan_recommendation,
    get_plan_details,
    escalate_to_support,
]

llm = ChatGroq(model=settings.groq_model, temperature=0.0)

lauki_agent = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)