"""Retry only a rolled-back SQL deadlock, never paid work or notifications."""
import random
import time


def transaction(conn, operation, attempts=3, pause=time.sleep):
    for attempt in range(attempts):
        conn.begin()
        try:
            value=operation()
            conn.commit()
            return value
        except BaseException as exc:
            conn.rollback()
            # InnoDB 1213 guarantees transaction rollback. Unknown commit /
            # connection failures must be reconciled by the existing item ID.
            if not getattr(exc,'args',None) or exc.args[0]!=1213 or attempt+1>=attempts:
                raise
            pause(random.uniform(.02,.08)*(attempt+1))
