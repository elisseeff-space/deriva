import duckdb

conn = duckdb.connect('deriva/adapters/database/sql.db')

# Simplified instruction for ApplicationComponent
simple_instruction = '''<persona>
You are a senior enterprise architect identifying application components from code repositories.
</persona>

<task>
Identify ApplicationComponents from candidate directories. An ApplicationComponent is a deployable module containing cohesive functionality.
</task>

<rules>
INCLUDE directories that:
- Contain source code files
- Represent cohesive functional areas
- Are not test, config, or build artifact directories

EXCLUDE:
- __pycache__, node_modules, dist, build
- tests/, test/, docs/
- Empty directories

NAMING:
- identifier: ac_<name> (snake_case)
- name: Title Case descriptive name
- documentation: Brief description of responsibility
</rules>

<output>
Return JSON with "elements" array containing:
- identifier: ac_<functional_name>
- name: Title Case name
- documentation: One sentence description
- source: Source node ID
- confidence: 0.7-1.0
</output>
'''

conn.execute("""
    UPDATE derivation_config 
    SET instruction = ?
    WHERE step_name = 'ApplicationComponent'
""", [simple_instruction])
conn.commit()

# Verify
result = conn.execute('''
    SELECT step_name, instruction 
    FROM derivation_config 
    WHERE step_name = 'ApplicationComponent'
''').fetchall()
print('Updated ApplicationComponent instruction:')
print(result[0][1][:500])

conn.close()
