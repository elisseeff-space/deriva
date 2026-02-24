import logging
logging.basicConfig(level=logging.DEBUG)

from deriva.services.session import PipelineSession
from deriva.modules.derivation.application_component import ApplicationComponentDerivation
from deriva.adapters.llm import LLMManager
from deriva.modules.derivation.base import Candidate, batch_candidates, build_derivation_prompt, parse_derivation_response, extract_response_content

with PipelineSession() as session:
    gm = session._graph_manager
    am = session._archimate_manager
    llm = session._llm_manager
    
    # Query candidates
    query = '''MATCH (n:Graph:Directory)
WHERE n.active = true
  AND NOT n.name IN ['__pycache__', 'node_modules', '.git', 'tests', 'test', 'docs', 'migrations']
RETURN n.id as id,
       COALESCE(n.name, n.path) AS name,
       n.path AS path,
       n.pagerank AS pagerank,
       n.louvain_community AS louvain_community,
       n.kcore_level AS kcore_level,
       n.is_articulation_point AS is_articulation_point'''
    
    rows = gm.query(query)
    print(f'Raw candidates: {len(rows)}')
    
    # Convert to Candidate objects
    candidates = []
    for r in rows:
        c = Candidate(
            node_id=r.get('id', ''),
            name=r.get('name', ''),
            pagerank=r.get('pagerank', 0),
            louvain_community=r.get('louvain_community'),
            kcore_level=r.get('kcore_level', 0),
            is_articulation_point=r.get('is_articulation_point', False),
            properties=r
        )
        candidates.append(c)
    
    # Get config
    configs = session.get_derivation_configs()
    config = None
    for c in configs:
        if c.get('step_name') == 'ApplicationComponent':
            config = c
            break
    
    if not config:
        print('Config not found!')
        exit(1)
    
    print(f'Config instruction: {config.get("instruction", "")[:100]}...')
    print(f'Config example: {config.get("example", "")[:100]}...')
    
    # Build prompt
    instruction = config.get('instruction', '')
    example = config.get('example', '')
    
    prompt = build_derivation_prompt(
        candidates=candidates,
        instruction=instruction,
        example=example,
        element_type='ApplicationComponent'
    )
    
    print(f'\n--- PROMPT ---\n{prompt[:500]}...')
    
    # Call LLM
    from deriva.modules.derivation.base import DERIVATION_SCHEMA
    response = llm.query(prompt, schema=DERIVATION_SCHEMA)
    
    print(f'\n--- RESPONSE ---')
    print(f'Type: {type(response)}')
    if hasattr(response, 'content'):
        print(f'Content: {response.content[:500]}...')
    
    # Parse
    content = response.content if hasattr(response, 'content') else str(response)
    parse_result = parse_derivation_response(content)
    print(f'\n--- PARSE RESULT ---')
    print(f'Success: {parse_result["success"]}')
    print(f'Data: {parse_result.get("data", [])}')
    print(f'Errors: {parse_result.get("errors", [])}')
