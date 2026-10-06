"""Additive installer. No deployment or enrollment occurs on module import."""
from pathlib import Path

TABLES = {
    'shared_kitchen':'owner_id', 'shared_archive_kitchen':'owner_id',
    'shared_shopping_list':'owner_id', 'shared_saved_recipes':'owner_id',
    'shared_dishes':'owner_id', 'shared_meal_calendar':'owner_id',
    'kitchen_amount_operations':'actor', 'user_dietary_preferences':'owner_id',
    'family_brain_subjects':'actor_id',
}


def touch(expression):
    return f"""INSERT INTO household_brain_state(household_id,enabled)
      SELECT {expression},default_serving FROM household_brain_config
      WHERE singleton=1 AND capture_enabled=1 AND {expression} IS NOT NULL
      ON DUPLICATE KEY UPDATE input_revision=input_revision+1,dirty=1,updated_at=UTC_TIMESTAMP(6);"""


def triggers():
    result = {}
    for table, column in TABLES.items():
        for operation, prefix in [('INSERT','i'),('UPDATE','u'),('DELETE','d')]:
            bodies = []
            for side in (['OLD','NEW'] if operation=='UPDATE' else ['OLD' if operation=='DELETE' else 'NEW']):
                # Dietary records are household keyed, even when the owner is not a
                # member. Other rows must resolve to a current member.
                expression = f'{side}.{column}' if table=='user_dietary_preferences' else (
                    f"(SELECT COALESCE(NULLIF(u.owner_id,''),u.user_id) FROM new_users u WHERE u.user_id={side}.{column})")
                bodies.append(touch(expression))
            name = f'hb_{prefix}_{table}'
            result[name] = f'CREATE TRIGGER {name} AFTER {operation} ON {table} FOR EACH ROW BEGIN ' + '\n'.join(bodies) + ' END'
    for operation, prefix in [('INSERT','i'),('UPDATE','u'),('DELETE','d')]:
        sides = ['OLD','NEW'] if operation=='UPDATE' else ['OLD' if operation=='DELETE' else 'NEW']
        name = 'hb_'+prefix+'_new_users'
        result[name] = f'CREATE TRIGGER {name} AFTER {operation} ON new_users FOR EACH ROW BEGIN '+ '\n'.join(
            touch(f"COALESCE(NULLIF({side}.owner_id,''),{side}.user_id)") for side in sides)+' END'
    return result


def install(conn):
    # Preflight ALL required canonical tables before nontransactional MySQL DDL.
    with conn.cursor() as cur:
        cur.execute('SHOW TABLES')
        existing = {next(iter(row.values())) for row in cur.fetchall()}
        required = set(TABLES) - {'family_brain_subjects'} | {'new_users'}
        if missing := required-existing:
            raise ValueError('missing_source_tables:'+','.join(sorted(missing)))
        for path in [Path(__file__).parent.parent/'family_brain/schema.sql',Path(__file__).with_name('schema.sql')]:
            for statement in path.read_text().split(';'):
                if statement.strip():
                    cur.execute(statement)
        cur.execute('SHOW TRIGGERS')
        existing_triggers = {r['Trigger']:r for r in cur.fetchall()}
        for name, ddl in triggers().items():
            if name in existing_triggers:
                expected = ddl.split(' FOR EACH ROW ',1)[1]
                if ' '.join(existing_triggers[name]['Statement'].split()) != ' '.join(expected.split()):
                    raise ValueError('trigger_definition_mismatch:'+name)
            else:
                cur.execute(ddl)
    conn.commit()


def provision(conn):
    """Reconcile preexisting households. Safe repeat; serving remains separately controlled."""
    with conn.cursor() as cur:
        cur.execute("""INSERT IGNORE INTO household_brain_state(household_id,enabled)
          SELECT DISTINCT COALESCE(NULLIF(u.owner_id,''),u.user_id),c.default_serving
          FROM new_users u JOIN household_brain_config c ON c.singleton=1 AND c.capture_enabled=1""")
    conn.commit()
