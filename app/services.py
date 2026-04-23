from sqlalchemy.orm import Session
from .models import Rep, PayoutCurve
from .formatter import format_currency, format_number, format_percentage
from .llm import generate_response
import re


def extract_rep_id_from_question(question: str):
    match = re.search(r"rep\s*-?\s*(\d+)", question.lower())
    if match:
        return match.group(1)
    return None


def find_applicable_rule(db, table, value, column):
    rule = db.query(table)\
        .filter(column <= value)\
        .order_by(column.desc())\
        .first()
    return rule


def get_payout_band(db, attainment):
    return find_applicable_rule(db, PayoutCurve, attainment, PayoutCurve.attainment_rate)


def get_rep_data(rep_id: str, db: Session):
    rep = db.query(Rep).filter(Rep.rep_id == rep_id).first()
    if not rep:
        return None

    attainment = rep.goal_achievement_rate * 100
    payout_row = get_payout_band(db, attainment)
    payout_rate = payout_row.payout_rate if payout_row else None

    data = {}
    for column in Rep.__table__.columns:
        value = getattr(rep, column.name)
        if hasattr(value, "quantize"):
            value = float(value)
        data[column.name] = value

    data["payout_rate"] = payout_rate
    return data


def detect_requested_field(question, rep_data):
    question = question.lower()
    for field in rep_data.keys():
        readable = field.replace("_", " ")
        if readable in question:
            return field
    return None


def build_rep_context(rep_data):
    return f"""
Representative Information
Name: {rep_data['rep_name']}
Role: {rep_data['role']}
Area: {rep_data['area_name']}
Region: {rep_data['region_name']}
Territory: {rep_data['territory_name']}

Performance
Total Goal TRx: {format_number(rep_data['total_trx_goal'])}
Quarter Goal TRx: {format_number(rep_data['qtd_trx_goal'])}
Quarter Achieved TRx: {format_number(rep_data['qtd_trx'])}
Goal Achievement Rate: {format_percentage(rep_data['goal_achievement_rate'])}

Commission
Projected Incremental TRx: {format_number(rep_data['total_projected_incremental_trx'])}
Commission Rate: {format_currency(rep_data['commission_rate'])}
Commission Earnings Value: {format_currency(rep_data['commission_earnings_value'])}

Totals
Total IC Earnings: {format_currency(rep_data['total_ic_earnings'])}
Total IC Payout: {format_currency(rep_data['total_ic_payout'])}

Eligibility
New Hire Eligibility: {rep_data['new_hire_eligibility']}
IC Eligibility: {rep_data['ic_eligibility']}

Payout Rate: {format_percentage(rep_data['payout_rate'])}
"""


def retrieve_policy_context(question, rag):
    if not rag:
        return None

    result = rag.ask(question)         

    if not isinstance(result, dict):   
        return None

    context = result.get("context")    

    if not context:
        return None

    return context


def detect_intent(question: str):
    prompt = f"""
Classify the user question into ONE category.

Categories:
greeting
rep_data
policy

Return ONLY one word.

Question:
{question}
"""
    intent_raw = generate_response(prompt).strip().lower()
    if "greeting" in intent_raw:
        return "greeting"
    elif "policy" in intent_raw:
        return "policy"
    else:
        return "rep_data"


def get_rep_explanation(rep_id: str, question: str, db: Session, rag):
    rep_data = get_rep_data(rep_id, db)

    mentioned_rep = extract_rep_id_from_question(question)
    if mentioned_rep and mentioned_rep != str(rep_id):
        return "Access denied. You can only view your own compensation data."

    if not rep_data:
        return "Rep not found."

    # Simple field answers
    field = detect_requested_field(question, rep_data)
    if field:
        value = rep_data.get(field)
        if value is None:
            return "The requested value is not available."
        if "payout" in field or "earnings" in field:
            value = format_currency(value)
        elif "rate" in field:
            value = format_percentage(value)
        readable = field.replace("_", " ")
        return f"Your {readable} is {value}."

    intent = detect_intent(question)

    if intent == "greeting":
        prompt = f"""
You are an ProcDNA IC support bot.
The representative name is {rep_data['rep_name']}.
Respond with a short greeting only.
Rules:
- Maximum 1 sentence
- Do not explain anything
- Do not add extra details

Example:
Hello {rep_data['rep_name']}, how can I help you today?
"""
        return generate_response(prompt)

    if intent == "policy":
        policy_context = retrieve_policy_context(question, rag)
        prompt = f"""
You are an IC compensation policy assistant.
Use ONLY the policy definitions below.

Policy Definitions:
{policy_context or "No policy context available."}

Question:
{question}
"""
        return generate_response(prompt)

    # Explanation / rep_data intent
    rep_context = build_rep_context(rep_data)
    policy_context = retrieve_policy_context(question, rag)

    prompt = f"""
You are an ProcDNA IC support bot, helping representative {rep_data['rep_name']}.

Rules:
- Never say you are an AI
- Never mention Microsoft, OpenAI, or any model
- Keep responses under 3 sentences unless explanation is explicitly requested
- Use ONLY the numbers in the representative data
- Do not invent numbers

Representative Data:
{rep_context}

Policy Definitions:
{policy_context or "No policy context available."}

User Question:
{question}
"""
    return generate_response(prompt)