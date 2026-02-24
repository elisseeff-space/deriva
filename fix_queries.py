import duckdb

conn = duckdb.connect('deriva/adapters/database/sql.db')

# List of all derivation steps
steps = [
    'ApplicationComponent', 'ApplicationService', 'ApplicationInterface',
    'DataObject', 'BusinessProcess', 'BusinessObject', 'BusinessFunction',
    'BusinessEvent', 'BusinessActor', 'TechnologyService', 'Node',
    'Device', 'SystemSoftware'
]

# Fix patterns:
# `Graph:Type` -> :Graph:Type
# n::GraphType` -> n:Graph:Type

for step in steps:
    result = conn.execute('''
        SELECT step_name, input_graph_query 
        FROM derivation_config 
        WHERE step_name = ?
    ''', [step]).fetchall()
    
    if not result:
        continue
    
    query = result[0][1]
    original = query
    
    # Fix pattern 1: `Graph:Type` -> :Graph:Type
    query = query.replace('`Graph:', ':Graph:')
    query = query.replace('`', '')  # Remove any remaining backticks
    
    # Fix pattern 2: n::GraphType` -> n:Graph:Type  
    query = query.replace('::', ':')
    
    if original != query:
        conn.execute('''
            UPDATE derivation_config 
            SET input_graph_query = ?
            WHERE step_name = ?
        ''', [query, step])
        print(f'Fixed {step}')
        print(f'  New: {query[:100]}...')
    else:
        print(f'No change for {step}')

conn.commit()

# Verify all
print('\n--- Verification ---')
for step in steps:
    result = conn.execute('''
        SELECT input_graph_query 
        FROM derivation_config 
        WHERE step_name = ?
    ''', [step]).fetchall()
    if result:
        q = result[0][0]
        has_backtick = '`' in q
        has_double_colon = '::' in q
        if has_backtick or has_double_colon:
            print(f'ISSUE: {step} - backtick={has_backtick}, double_colon={has_double_colon}')
        else:
            print(f'OK: {step}')

conn.close()
