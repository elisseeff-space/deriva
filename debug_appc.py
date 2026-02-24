from deriva.services.session import PipelineSession
from deriva.modules.derivation.application_component import ApplicationComponentDerivation
from deriva.modules.derivation.base import Candidate

with PipelineSession() as session:
    gm = session._graph_manager
    am = session._archimate_manager
    llm = session._llm_manager
    
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
    
    # Query candidates
    query = config['input_graph_query']
    rows = gm.query(query)
    print(f'Raw candidates: {len(rows)}')
    for r in rows:
        print(f'  - {r.get("id")}: {r.get("name")}')
    
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
    
    # Get enrichments
    from deriva.modules.derivation.base import get_enrichments_from_neo4j
    enrichments = get_enrichments_from_neo4j(gm, config_name='ApplicationComponent')
    
    # Filter
    derivation = ApplicationComponentDerivation()
    filtered = derivation.filter_candidates(candidates, enrichments, max_candidates=30)
    print(f'\nFiltered candidates: {len(filtered)}')
    for c in filtered:
        print(f'  - {c.node_id}: {c.name}')
    
    # Check existing elements for duplicate filtering
    existing = am.get_elements(element_type='ApplicationComponent', enabled_only=True)
    print(f'\nExisting ApplicationComponent elements: {len(existing)}')
    
    # Test _filter_existing_duplicates
    filtered_after_dup = derivation._filter_existing_duplicates(filtered, am)
    print(f'\nAfter duplicate filter: {len(filtered_after_dup)}')
    for c in filtered_after_dup:
        print(f'  - {c.node_id}: {c.name}')
