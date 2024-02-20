# distributed_jobs/operations.py
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from contextlib import contextmanager
from .models import Base, Job
import datetime

# Initialize variables for engine and session
engine = None
Session = None

def configure_database(database_engine,reset=False):
    """Configure the database engine and create a session factory."""
    global engine, Session
    engine = database_engine
    Base.metadata.bind = engine  # Bind the engine to the Base metadata
    Session = sessionmaker(bind=engine)

    if reset:
        Base.metadata.drop_all(engine)

    Base.metadata.create_all(engine)

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    global Session
    if Session is None:
        raise Exception("Database has not been configured. Call configure_database first.")
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def add_job(payload=None, cid=None):
    with session_scope() as session:
        new_job = Job(state='pending',cid=cid, payload=payload)
        session.add(new_job)

def add_jobs(jobs_list):
    with session_scope() as session:
        session.add_all(jobs_list)

def fetch_job(return_mode='job'):
    with session_scope() as session:
        job = session.query(Job).filter_by(state='pending').with_for_update().first()
        if job:
            job.state = 'in_progress'

            if return_mode == 'job':
                return job
            elif return_mode == 'payload':
                return job.payload
            elif return_mode == 'both':
                return job, job.payload
            else:
                return None

def process_job(job_func,**kwargs):
    job,job_payload = fetch_job(return_mode='both')

    if not job:
        return "No job available"
    
    #track the time the process takes
    start_time = datetime.datetime.now()

    try:
        job_func(**{**kwargs, **job_payload})
        job.state = 'completed'
    except Exception as e:
        job.state = 'failed'
        job.traceback = str(e)
    finally:
        end_time = datetime.datetime.now()
        job.duration = end_time - start_time

        with session_scope() as session:
            session.add(job)

def job_statistics():
    with session_scope() as session:
        stats = {
            'running': session.query(Job).filter_by(state='in_progress').count(),
            'pending': session.query(Job).filter_by(state='pending').count(),
            'completed': session.query(Job).filter_by(state='completed').count(),
            'failed': session.query(Job).filter_by(state='failed').count(),
            'freezed': session.query(Job).filter_by(state='freezed').count(),
        }
        return stats

def get_jobs_by_state(state,return_payload=False):
    with session_scope() as session:
        jobs = session.query(Job).filter_by(state=state).all()
        if return_payload:
            jobs = [(job.cid, job.payload) for job in jobs]
        else:
            jobs = [job.cid for job in jobs]
        return jobs
    

def fetch_jobs_with_traceback():
    with session_scope() as session:
        # Query jobs where traceback is not None
        jobs_with_traceback = session.query(Job).filter(Job.traceback.isnot(None)).all()
        
        # Extracting custom_identifier and traceback for each job
        result = [(job.cid, job.traceback) for job in jobs_with_traceback]
        
        return result
    
def retry_failed_jobs():
    """
    Switch the state of failed jobs back to pending, allowing them to be retried.
    """
    with session_scope() as session:
        # Query for failed jobs
        failed_jobs = session.query(Job).filter_by(state='failed').all()
        
        # Update state of each failed job to pending
        for job in failed_jobs:
            job.state = 'pending'


def retry_failed_job_by_identifier(custom_identifier):
    """
    Switch the state of a failed job with the given custom_identifier back to pending, allowing it to be retried.
    
    Args:
    custom_identifier: The unique identifier of the job to be retried.
    """
    with session_scope() as session:
        # Query for a failed job with the given custom_identifier
        job = session.query(Job).filter_by(custom_identifier=custom_identifier, state='failed').first()
        
        if job:
            # Update the job's state to pending
            job.state = 'pending'
        else:
            # If no job matches the criteria, raise an exception or handle it as needed
            print(f"No failed job found with custom_identifier: {custom_identifier}")
            # Optionally, you can raise an exception if the job is not found
            # raise ValueError(f"No failed job found with custom_identifier: {custom_identifier}")

def freeze_pending_jobs():
    """
    Switch the state of pending jobs to freezed, preventing them from being processed.
    """
    with session_scope() as session:
        # Query for pending jobs
        pending_jobs = session.query(Job).filter_by(state='pending').all()
        
        # Update state of each pending job to freezed
        for job in pending_jobs:
            job.state = 'freezed'

def freeze_pending_job_by_identifier(custom_identifier):
    """
    Switch the state of a pending job with the given custom_identifier to freezed, preventing it from being processed.
    
    Args:
    custom_identifier: The unique identifier of the job to be freezed.
    """
    with session_scope() as session:
        # Query for a pending job with the given custom_identifier
        job = session.query(Job).filter_by(custom_identifier=custom_identifier, state='pending').first()
        
        if job:
            # Update the job's state to freezed
            job.state = 'freezed'
        else:
            # If no job matches the criteria, raise an exception or handle it as needed
            print(f"No pending job found with custom_identifier: {custom_identifier}")
            # Optionally, you can raise an exception if the job is not found
            # raise ValueError(f"No pending job found with custom_identifier: {custom_identifier}")

def unfreeze_freezed_jobs():
    """
    Switch the state of freezed jobs back to pending, allowing them to be processed.
    """
    with session_scope() as session:
        # Query for freezed jobs
        freezed_jobs = session.query(Job).filter_by(state='freezed').all()
        
        # Update state of each freezed job to pending
        for job in freezed_jobs:
            job.state = 'pending'

def unfreeze_freezed_job_by_identifier(custom_identifier):
    """
    Switch the state of a freezed job with the given custom_identifier back to pending, allowing it to be processed.
    
    Args:
    custom_identifier: The unique identifier of the job to be unfreezed.
    """
    with session_scope() as session:
        # Query for a freezed job with the given custom_identifier
        job = session.query(Job).filter_by(custom_identifier=custom_identifier, state='freezed').first()
        
        if job:
            # Update the job's state to pending
            job.state = 'pending'
        else:
            # If no job matches the criteria, raise an exception or handle it as needed
            print(f"No freezed job found with custom_identifier: {custom_identifier}")
            # Optionally, you can raise an exception if the job is not found
            # raise ValueError(f"No freezed job found with custom_identifier: {custom_identifier}")


def delete_job_by_identifier(custom_identifier):
    """
    Delete a job with the given custom_identifier from the database.
    
    Args:
    custom_identifier: The unique identifier of the job to be deleted.
    """
    with session_scope() as session:
        # Query for a job with the given custom_identifier
        job = session.query(Job).filter_by(custom_identifier=custom_identifier).first()
        
        if job:
            # Delete the job
            session.delete(job)
        else:
            # If no job matches the criteria, raise an exception or handle it as needed
            print(f"No job found with custom_identifier: {custom_identifier}")
            # Optionally, you can raise an exception if the job is not found
            # raise ValueError(f"No job found with custom_identifier: {custom_identifier}")

def fetch_long_running_jobs(threshold):
    """Fetch jobs that took longer than a specified threshold to run.

    Args:
        threshold (datetime.timedelta): The duration threshold for determining long-running jobs.

    Returns:
        list: A list of Job objects that exceeded the threshold.
    """
    with session_scope() as session:
        long_running_jobs = session.query(Job) \
                                   .filter(Job.state == 'completed') \
                                   .filter(Job.duration > threshold) \
                                   .all()
        return long_running_jobs


def analyze_job_durations():
    """Analyzes job durations and provides relevant statistics."""
    with session_scope() as session:
        completed_jobs = session.query(Job).filter_by(state='completed').all()
    durations = [job.duration for job in completed_jobs]

    if durations:
        import numpy as np  # You'll need numpy for these calculations
        average_duration = np.mean(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)

        return {
            'average_duration': average_duration,
            'min_duration': min_duration,
            'max_duration': max_duration
        }
    else:
        return None  # No completed jobs for duration analysis

def print_statistics(job_stats, duration_stats):
    """Presents the calculated statistics in a user-friendly format."""
    print("\n---- Job Statistics ----")
    for state, count in job_stats.items():
        print(f"{state.capitalize()}: {count}")

    if duration_stats:
        print("\n---- Job Duration Statistics ----")
        for metric, value in duration_stats.items():
            print(f"{metric.replace('_', ' ')}: {value}")
    else:
        print("\n---- Job Duration Statistics ----")
        print("No completed jobs found for duration analysis.")
        print("\n")

def print_summary():
    
    job_stats = job_statistics()
    duration_stats = analyze_job_durations()
    print_statistics(job_stats, duration_stats)