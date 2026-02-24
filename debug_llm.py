from deriva.services.session import PipelineSession
from deriva.modules.derivation.base import Candidate, build_derivation_prompt, parse_derivation_response, DERIVATION_SCHEMA
from deriva.adapters.llm import LLMManager

with PipelineSession() as session:
    gm = session._graph_manager
    llm = LLMManager()  # Create LLM manager directly
    
    # Get config
    configs = session.get_derivation_configs()
    config = None
    for c in configs:
        if c.get('step_name') == 'ApplicationComponent':
            config = c
            break
    
    # Query candidates
    query = config['input_graph_query']
    rows = gm.query(query)
    
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
    
    # Build prompt
    instruction = config['instruction']
    example = config['example']
    
    prompt = build_derivation_prompt(
        candidates=candidates,
        instruction=instruction,
        example=example,
        element_type='ApplicationComponent'
    )
    
    print('=== PROMPT ===')
    print(prompt[:2000])
    print('...\n')
    
    # Call LLM
    response = llm.query(prompt, schema=DERIVATION_SCHEMA)
    
    print('=== RESPONSE ===')
    content = response.content if hasattr(response, 'content') else str(response)
    print(content)
    print()
    
    # Parse
    parse_result = parse_derivation_response(content)
    print('=== PARSE RESULT ===')
    print(f'Success: {parse_result["success"]}')
    print(f'Data count: {len(parse_result.get("data", []))}')
    for d in parse_result.get('data', []):
        print(f'  - {d.get("identifier")}: {d.get("name")}')
    print(f'Errors: {parse_result.get("errors", [])}')
