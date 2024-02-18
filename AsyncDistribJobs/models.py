from sqlalchemy import create_engine, Column, Integer, String, JSON, Interval
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Job(Base):
    __tablename__ = 'jobs'
    id = Column(Integer, primary_key=True)
    state = Column(String, index=True)
    cid = Column(String)
    payload = Column(JSON)
    traceback = Column(String)
    duration = Column(Interval)