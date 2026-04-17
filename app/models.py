from sqlalchemy import Column, String, Integer, Numeric, DateTime, Text, ForeignKey, Boolean
from .db import Base 
from datetime import datetime
from sqlalchemy.orm import relationship


class Rep(Base):
    __tablename__ = "reps"

    area_name = Column(String)
    area_id = Column(String)
    region_id = Column(String)
    region_name = Column(String)
    territory_id = Column(String)
    territory_name = Column(String)
    role = Column(String)

    rep_id = Column(String, primary_key=True)
    rep_name = Column(String)

    no_of_underlying_regions = Column(Integer)
    no_of_underlying_tss_territories = Column(Integer)
    no_of_underlying_overall_territories = Column(Integer)

    total_trx_goal = Column(Numeric)
    qtd_trx_goal = Column(Numeric)
    qtd_trx = Column(Numeric)
    qtd_trx_per_territory = Column(Numeric)

    goal_achievement_rate = Column(Numeric)
    ic_earnings_rate = Column(Numeric)

    target_pay = Column(Numeric)
    ic_earnings_value = Column(Numeric)

    total_projected_incremental_trx = Column(Numeric)

    commission_rate = Column(Numeric)
    commission_earnings_value = Column(Numeric)

    total_ic_earnings = Column(Numeric)
    qtd_ic_earnings_rate = Column(Numeric)

    new_hire_eligibility = Column(Integer)
    ic_eligibility = Column(Integer)

    total_ic_payout = Column(Numeric)

class PayoutCurve(Base):
    __tablename__ = "payout_curve"

    id = Column(Integer, primary_key=True)
    attainment_rate = Column(Numeric)
    payout_rate = Column(Numeric)
    tbm_target_earnings = Column(Numeric)
    rbd_target_earnings = Column(Numeric)
    abd_target_earnings = Column(Numeric)  

class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id = Column(String, primary_key=True, index=True)
    rep_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    escalated = Column(Boolean, default=False)
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.conversation_id"))
    role = Column(String)
    message_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")