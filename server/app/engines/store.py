# app/engines/store.py
# Shared in-memory stores — imported by both routes and engines
# to avoid circular imports

file_store = {}   # moved here as a backup reference — upload.py still owns this
run_store  = {}   # pipeline.py writes run_id, executor.py updates progress