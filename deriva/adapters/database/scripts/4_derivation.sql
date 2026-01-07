-- Derivation Config Seed Data
-- Phase: prep (graph algorithms), generate (LLM derivation), refine (validation), relationship (per-element relationships)
-- Sequence determines execution order within each phase

-- =============================================================================
-- PREP PHASE: Graph algorithms that filter/prepare nodes before LLM derivation
-- =============================================================================

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, params, is_active)
VALUES (1, 'k_core_filter', 'prep', 1, 1, TRUE, FALSE,
    'MATCH (n:Graph) OPTIONAL MATCH (n)-[r]-() WITH n, count(r) as degree RETURN n, degree',
    '{"k": 2, "description": "Remove nodes with degree < k"}', TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, params, is_active)
VALUES (2, 'scc_detection', 'prep', 1, 2, FALSE, FALSE,
    'MATCH (n:Graph)-[r]->(m:Graph) RETURN n, r, m',
    '{"min_size": 2, "description": "Detect strongly connected components"}', TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, params, is_active)
VALUES (3, 'louvain_communities', 'prep', 1, 3, FALSE, FALSE,
    'MATCH (n:Graph)-[r]->(m:Graph) RETURN n, r, m',
    '{"resolution": 1.0, "description": "Detect communities using Louvain algorithm"}', TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, params, is_active)
VALUES (4, 'articulation_points', 'prep', 1, 4, FALSE, FALSE,
    'MATCH (n:Graph)-[r]-(m:Graph) RETURN n, r, m',
    '{"mark_critical": true, "description": "Identify articulation points (bridge nodes)"}', TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, params, is_active)
VALUES (5, 'pagerank', 'prep', 1, 5, TRUE, FALSE,
    'MATCH (n:Graph)-[r]->(m:Graph) RETURN n, r, m',
    '{"damping": 0.85, "iterations": 20, "description": "Calculate node significance via PageRank"}', TRUE);

-- =============================================================================
-- GENERATE PHASE: LLM-based element derivation from graph nodes
-- =============================================================================

-- ApplicationComponent
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (101, 'ApplicationComponent', 'generate', 1, 1, TRUE, TRUE,
'MATCH (n:`Graph:Directory`)
WHERE n.active = true
  AND NOT n.name IN [''__pycache__'', ''node_modules'', ''.git'', ''.venv'', ''venv'', ''dist'', ''build'',
                     ''static'', ''assets'', ''public'', ''images'', ''img'', ''css'', ''js'', ''fonts'',
                     ''templates'', ''views'', ''layouts'', ''partials'']
  AND NOT n.path =~ ''.*(test|spec|__pycache__|node_modules|\\.git|\\.venv|venv|dist|build).*''
RETURN n.id as id, n.name as name, labels(n) as labels, properties(n) as properties',
'You are identifying ApplicationComponent elements from source code directories.

An ApplicationComponent is a modular, deployable part of a system that:
- Encapsulates related functionality (not just a folder)
- Has clear boundaries and responsibilities
- Contains code that works together as a unit
- Could potentially be a separate module or package

Each candidate includes graph metrics to help assess importance:
- pagerank: How central/important the directory is
- community: Which cluster of related code it belongs to
- kcore: How connected it is to the core codebase
- is_bridge: Whether it connects different parts of the codebase

Review each candidate and decide which should become ApplicationComponent elements.

INCLUDE directories that:
- Represent cohesive functional units (services, modules, packages)
- Have meaningful names indicating purpose
- Are structural roots of related code

EXCLUDE directories that:
- Are just organizational containers with no cohesive purpose
- Contain only configuration or static assets
- Are too granular (single-file directories)',
'{
  "elements": [
    {
      "identifier": "appcomp_user_service",
      "name": "User Service",
      "documentation": "Handles user authentication, registration, and profile management",
      "source": "dir_myproject_src_services_user",
      "confidence": 0.9
    },
    {
      "identifier": "appcomp_frontend",
      "name": "Frontend Application",
      "documentation": "React-based web interface for the application",
      "source": "dir_myproject_frontend",
      "confidence": 0.85
    }
  ]
}', 30, 10, TRUE);

-- ApplicationService
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (102, 'ApplicationService', 'generate', 1, 2, TRUE, TRUE,
'MATCH (n:`Graph:Method`)
WHERE n.active = true
RETURN n.id as id,
       COALESCE(n.name, n.methodName) as name,
       labels(n) as labels,
       properties(n) as properties',
'You are identifying ApplicationService elements from source code methods.

An ApplicationService represents explicitly exposed application behavior:
- Web routes and API endpoints
- Service interfaces that external clients can call
- Handlers that respond to external requests

Each candidate includes method information and graph metrics.

Review each candidate and decide which should become ApplicationService elements.

INCLUDE methods that:
- Handle HTTP requests (routes, endpoints, views)
- Expose functionality to external clients
- Are entry points for user interactions
- Have names suggesting they respond to requests

EXCLUDE methods that:
- Are internal/private helpers
- Are utility functions
- Are lifecycle methods (__init__, setup, etc.)
- Only perform internal processing

When naming:
- Use service-oriented names (e.g., "Invoice Form Service" not "invoice_form")
- Describe what the service provides',
'{
  "elements": [
    {
      "identifier": "as_invoice_form",
      "name": "Invoice Form Service",
      "documentation": "Web endpoint for creating and managing invoice data through a form interface",
      "source": "method_invoice_form",
      "confidence": 0.9
    },
    {
      "identifier": "as_export_pdf",
      "name": "PDF Export Service",
      "documentation": "Endpoint for generating and downloading invoice PDFs",
      "source": "method_invoice_pdf",
      "confidence": 0.85
    }
  ]
}', 30, 10, TRUE);

-- ApplicationInterface
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (103, 'ApplicationInterface', 'generate', 1, 3, TRUE, TRUE,
    'MATCH (f:Graph:File) WHERE f.fileType=''source'' AND f.subtype IN [''python'',''javascript'',''typescript''] AND f.active = true RETURN f',
    'Identify externally exposed interfaces (REST, gRPC, CLI). Prefer files declaring routes or endpoints or RPC definitions. Summarize protocol and endpoint root.',
    '{"identifier":"if:rest","name":"REST API","protocol":"HTTP","endpoint":"/api","exposes":["POST /login","POST /logout"],"confidence":0.7}',
    30, 10, TRUE);

-- DataObject
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (104, 'DataObject', 'generate', 1, 4, TRUE, TRUE,
'MATCH (n:`Graph:File`)
WHERE n.active = true
RETURN n.id as id,
       COALESCE(n.fileName, n.name) as name,
       labels(n) as labels,
       properties(n) as properties',
'You are identifying DataObject elements from files in a codebase.

A DataObject represents data structured for automated processing:
- Database files (SQLite, SQL scripts)
- Configuration files (JSON, YAML, ENV)
- Schema definitions
- Data exchange formats

Each candidate includes file information and graph metrics.

Review each candidate and decide which should become DataObject elements.

INCLUDE files that:
- Store application data (databases, data files)
- Define configuration (settings, environment)
- Define data schemas or structures
- Are used for data exchange

EXCLUDE files that:
- Are source code (Python, JavaScript, etc.)
- Are templates (HTML, Jinja)
- Are documentation (README, docs)
- Are static assets (images, CSS)

When naming:
- Use descriptive names (e.g., "Application Database" not "database.db")
- Indicate the data''s purpose',
'{
  "elements": [
    {
      "identifier": "do_application_database",
      "name": "Application Database",
      "documentation": "SQLite database storing invoices, customers, and line items",
      "source": "file_database.db",
      "confidence": 0.95
    },
    {
      "identifier": "do_app_configuration",
      "name": "Application Configuration",
      "documentation": "Environment configuration for Flask application settings",
      "source": "file_.flaskenv",
      "confidence": 0.85
    }
  ]
}', 30, 10, TRUE);

-- BusinessProcess
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (105, 'BusinessProcess', 'generate', 1, 5, TRUE, TRUE,
'MATCH (n:`Graph:Method`)
WHERE n.active = true
RETURN n.id as id,
       COALESCE(n.name, n.methodName) as name,
       labels(n) as labels,
       properties(n) as properties',
'You are identifying BusinessProcess elements from source code methods.

A BusinessProcess represents a sequence of business behaviors that achieves
a specific outcome. It is NOT just any function - it represents a complete
business activity that delivers value.

Each candidate includes graph metrics to help assess importance:
- pagerank: How central/important the method is
- in_degree/out_degree: How connected it is

Review each candidate and decide which should become BusinessProcess elements.

INCLUDE methods that:
- Represent complete business activities (Create Invoice, Process Payment)
- Coordinate multiple steps to achieve a business outcome
- Would be meaningful to a business analyst
- Are named with verbs indicating business actions

EXCLUDE methods that:
- Are purely technical (validation, parsing, formatting)
- Are framework lifecycle methods (__init__, setup, etc.)
- Are simple getters/setters
- Are utility/helper functions
- Only do one small technical step

When naming:
- Use business-friendly verb phrases (e.g., "Create Invoice" not "create_invoice")
- Focus on the business outcome, not technical implementation',
'{
  "elements": [
    {
      "identifier": "bp_create_invoice",
      "name": "Create Invoice",
      "documentation": "Process of generating a new invoice with line items and customer details",
      "source": "method_invoice_form",
      "confidence": 0.9
    },
    {
      "identifier": "bp_process_payment",
      "name": "Process Payment",
      "documentation": "Handles payment submission and validation for customer orders",
      "source": "method_handle_payment",
      "confidence": 0.85
    }
  ]
}', 30, 10, TRUE);

-- BusinessObject
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (106, 'BusinessObject', 'generate', 1, 6, TRUE, TRUE,
'MATCH (n)
WHERE (n:`Graph:TypeDefinition` OR n:`Graph:BusinessConcept`)
  AND n.active = true
RETURN n.id as id,
       COALESCE(n.name, n.typeName, n.conceptName) as name,
       labels(n) as labels,
       properties(n) as properties',
'You are identifying BusinessObject elements from source code type definitions.

A BusinessObject represents a passive element that has business relevance:
- Data entities that the business cares about (Customer, Order, Invoice)
- Domain concepts that appear in business conversations
- Information structures that would appear in business documentation

Each candidate includes graph metrics to help assess importance:
- pagerank: How central/important the type is
- community: Which cluster of related types it belongs to
- in_degree: How many other types reference it (higher = more important)

Review each candidate and decide which should become BusinessObject elements.

INCLUDE types that:
- Represent real-world business concepts (Customer, Order, Product)
- Are data entities that store business information
- Would be meaningful to a business analyst (not just a developer)
- Have names that are nouns representing "things" the business cares about

EXCLUDE types that:
- Are purely technical (handlers, adapters, decorators)
- Are framework/library classes (BaseModel, FlaskForm)
- Are utility classes (StringHelper, DateUtils)
- Are internal implementation details
- Are exceptions or error types
- Are configuration or settings classes

When naming:
- Use business-friendly names (e.g., "Invoice" not "InvoiceModel")
- Capitalize appropriately (e.g., "Customer Order" not "customer_order")',
'{
  "elements": [
    {
      "identifier": "bo_invoice",
      "name": "Invoice",
      "documentation": "A commercial document issued by a seller to a buyer, indicating products, quantities, and prices",
      "source": "type_Invoice",
      "confidence": 0.95
    },
    {
      "identifier": "bo_customer",
      "name": "Customer",
      "documentation": "A person or organization that purchases goods or services",
      "source": "type_Customer",
      "confidence": 0.9
    },
    {
      "identifier": "bo_line_item",
      "name": "Line Item",
      "documentation": "An individual entry on an invoice representing a product or service with quantity and price",
      "source": "type_Position",
      "confidence": 0.85
    }
  ]
}', 30, 10, TRUE);

-- BusinessFunction
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (107, 'BusinessFunction', 'generate', 1, 7, TRUE, TRUE,
    'MATCH (b:Graph:BusinessConcept) WHERE b.conceptType=''function'' AND b.active = true RETURN b',
    'Identify business functions from domain concepts. Group related business behaviors by capability or responsibility area.',
    '{"identifier":"bf:payment","name":"Payment Processing","description":"Handles all payment-related operations","confidence":0.65}',
    30, 10, TRUE);

-- BusinessEvent
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (108, 'BusinessEvent', 'generate', 1, 8, TRUE, TRUE,
    'MATCH (b:Graph:BusinessConcept) WHERE b.conceptType=''event'' AND b.active = true RETURN b',
    'Map domain events to BusinessEvent elements. Focus on state changes that trigger business processes or decisions.',
    '{"identifier":"be:order-placed","name":"Order Placed","trigger":"Customer completes checkout","confidence":0.6}',
    30, 10, TRUE);

-- BusinessActor
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (109, 'BusinessActor', 'generate', 1, 9, TRUE, TRUE,
'MATCH (n)
WHERE (n:`Graph:TypeDefinition` OR n:`Graph:BusinessConcept`)
  AND n.active = true
RETURN n.id as id,
       COALESCE(n.name, n.typeName, n.conceptName) as name,
       labels(n) as labels,
       properties(n) as properties',
'You are identifying BusinessActor elements from source code types and concepts.

A BusinessActor represents a business entity capable of performing behavior:
- Users and roles (Customer, Administrator, Operator)
- Organizational units (Department, Team)
- External parties (Supplier, Partner)
- System actors when they represent a logical role

Each candidate includes graph metrics to help assess importance.

Review each candidate and decide which should become BusinessActor elements.

INCLUDE types that:
- Represent people, roles, or organizational entities
- Can initiate or perform business activities
- Would appear in a business context diagram
- Have names indicating actors (User, Customer, Manager, etc.)

EXCLUDE types that:
- Represent data/information (Invoice, Order, Report)
- Are technical components (Controller, Handler, Service)
- Are utility/framework classes
- Are abstract base classes

When naming:
- Use role names (e.g., "Customer" not "CustomerModel")
- Be specific about the actor''s function',
'{
  "elements": [
    {
      "identifier": "ba_customer",
      "name": "Customer",
      "documentation": "External party who purchases products or services and receives invoices",
      "source": "type_Customer",
      "confidence": 0.95
    },
    {
      "identifier": "ba_administrator",
      "name": "Administrator",
      "documentation": "Internal user with elevated privileges for system management",
      "source": "type_Admin",
      "confidence": 0.9
    }
  ]
}', 20, 10, TRUE);

-- TechnologyService
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (110, 'TechnologyService', 'generate', 1, 10, TRUE, TRUE,
'MATCH (n:`Graph:ExternalDependency`)
WHERE n.active = true
RETURN n.id as id,
       COALESCE(n.dependencyName, n.name) as name,
       labels(n) as labels,
       properties(n) as properties',
'You are identifying TechnologyService elements from external dependencies.

A TechnologyService represents an externally visible unit of functionality
provided by infrastructure or external systems, such as:
- Databases (PostgreSQL, MongoDB, Redis, etc.)
- Message queues (Kafka, RabbitMQ, etc.)
- External APIs and HTTP clients
- Cloud services (AWS S3, Azure Blob, etc.)
- Authentication services

Review each candidate dependency. Consider:
- Does this provide infrastructure functionality?
- Is it a service the application connects TO (not just a utility library)?
- Would it appear in an architecture diagram?

INCLUDE:
- Database drivers and ORMs (sqlalchemy, psycopg2, pymongo)
- HTTP clients for external APIs (requests, httpx, axios)
- Message queue clients (kafka-python, pika)
- Cloud SDK components (boto3, azure-storage)
- Caching services (redis, memcached)

EXCLUDE:
- Standard library modules
- Utility libraries (json parsing, date handling)
- Testing frameworks
- Development tools
- Internal application modules',
'{
  "elements": [
    {
      "identifier": "techsvc_postgresql",
      "name": "PostgreSQL Database",
      "documentation": "Relational database service for persistent data storage",
      "source": "dep_sqlalchemy",
      "confidence": 0.95
    },
    {
      "identifier": "techsvc_redis_cache",
      "name": "Redis Cache",
      "documentation": "In-memory data store used for caching and session management",
      "source": "dep_redis",
      "confidence": 0.9
    }
  ]
}', 30, 10, TRUE);

-- Node
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (111, 'Node', 'generate', 1, 11, TRUE, TRUE,
    'MATCH (e:Graph:ExternalDependency) WHERE e.category=''infrastructure'' AND e.active = true RETURN e',
    'Map infrastructure resources to Node elements. Include servers, containers, cloud resources that host or execute software.',
    '{"identifier":"node:app-server","name":"Application Server","type":"virtual","platform":"AWS EC2","confidence":0.75}',
    30, 10, TRUE);

-- Device
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (112, 'Device', 'generate', 1, 12, TRUE, TRUE,
    'MATCH (e:Graph:ExternalDependency) WHERE e.category=''hardware'' AND e.active = true RETURN e',
    'Identify physical devices from dependencies. Focus on hardware resources like servers, network equipment, or IoT devices.',
    '{"identifier":"dev:load-balancer","name":"Load Balancer","type":"network","vendor":"F5","confidence":0.8}',
    30, 10, TRUE);

-- SystemSoftware
INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, instruction, example, max_candidates, batch_size, is_active)
VALUES (113, 'SystemSoftware', 'generate', 1, 13, TRUE, TRUE,
    'MATCH (e:Graph:ExternalDependency) WHERE e.category IN [''runtime'',''platform''] AND e.active = true RETURN e',
    'Map system software and platforms to SystemSoftware elements. Include runtimes, operating systems, middleware, and platform services.',
    '{"identifier":"sys:nodejs","name":"Node.js Runtime","version":"18.x","category":"runtime","confidence":0.85}',
    30, 10, TRUE);

-- =============================================================================
-- REFINE PHASE: Validation and improvement of derived model
-- =============================================================================

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, input_model_query, instruction, example, is_active)
VALUES (201, 'Completeness', 'refine', 1, 1, TRUE, TRUE,
    'MATCH (n:Graph) WHERE n.significance > 0.5 AND n.active = true RETURN n',
    'MATCH (e:Model) RETURN e',
    'Compare graph nodes with ArchiMate elements. Identify graph nodes with high significance (>0.5) that don''t have corresponding ArchiMate elements. For each missing element, suggest creating an appropriate ArchiMate element with proper type, name, and relationships. Return an array of new elements to add with their confidence scores.',
    '{"new_elements":[{"identifier":"ac:payment-processor","type":"ApplicationComponent","name":"Payment Processor","source_node":"Graph:PaymentService","reason":"High-significance service node without ArchiMate representation","confidence":0.85}],"new_relationships":[]}',
    TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, input_model_query, instruction, example, is_active)
VALUES (202, 'RelationshipConsistency', 'refine', 1, 2, TRUE, TRUE,
    'MATCH (source:Graph)-[r]->(target:Graph) WHERE source.significance > 0.3 AND target.significance > 0.3 AND source.active = true AND target.active = true RETURN source, type(r) as rel_type, target',
    'MATCH (source:Model)-[r]->(target:Model) RETURN source, type(r) as rel_type, target',
    'Compare relationships between graph nodes and ArchiMate elements. Identify missing or incorrect relationships in the ArchiMate model. For each discrepancy, suggest adding or updating relationships using appropriate ArchiMate relationship types (Composition, Aggregation, Serving, Realization, Access, Flow). Ensure relationship types follow ArchiMate metamodel rules. Return arrays of new and updated relationships.',
    '{"new_elements":[],"new_relationships":[{"source":"ac:api-gateway","target":"as:user-service","type":"Serving","reason":"Graph shows API Gateway calls User Service, missing in ArchiMate","confidence":0.9}],"updated_relationships":[{"source":"ac:database","target":"do:user-data","old_type":"Composition","new_type":"Access","reason":"Database accesses data, not composes it","confidence":0.85}]}',
    TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, input_graph_query, input_model_query, instruction, example, is_active)
VALUES (203, 'LayeringAbstraction', 'refine', 1, 3, FALSE, TRUE,
    'MATCH (n:Graph) WHERE n.active = true RETURN n, labels(n) as node_labels, n.category as category',
    'MATCH (e:Model) RETURN e, e.type as element_type, e.layer as layer',
    'Analyze the ArchiMate model for proper layering and abstraction. Check: 1) Elements are in correct layers (Business/Application/Technology), 2) Abstraction levels are appropriate (not too technical in business layer, not too abstract in technology layer), 3) Cross-layer relationships follow ArchiMate patterns. Suggest reclassifying elements or splitting/merging elements to improve architectural clarity. Return arrays of reclassified elements, split suggestions, and new relationships.',
    '{"reclassified_elements":[{"identifier":"ac:user-authentication","current_type":"ApplicationComponent","suggested_type":"ApplicationService","reason":"Represents behavior/capability, not a structural component","confidence":0.8}],"split_suggestions":[{"identifier":"ac:monolith-app","suggested_elements":[{"type":"ApplicationComponent","name":"User Management Module"},{"type":"ApplicationComponent","name":"Order Processing Module"},{"type":"ApplicationComponent","name":"Inventory Module"}],"reason":"Large monolithic component should be decomposed for clarity","confidence":0.7}],"new_relationships":[]}',
    TRUE);

-- =============================================================================
-- RELATIONSHIP PHASE: Per-element-type relationship derivation
-- Uses ArchiMate metamodel to constrain valid relationship types
-- =============================================================================

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, instruction, example, is_active)
VALUES (301, 'ApplicationComponent_relationships', 'relationship', 1, 1, TRUE, TRUE,
    'Derive relationships FROM ApplicationComponent elements. Valid relationships: Composition (to other structure elements), Aggregation (to structure/behavior/passive), Assignment (to behavior elements like ApplicationService), Serving (to other elements), Access (to passive elements like DataObject).',
    '{"relationships":[{"source":"ac:auth","target":"as:login-service","relationship_type":"Assignment","confidence":0.85},{"source":"ac:auth","target":"do:user-data","relationship_type":"Access","confidence":0.8}]}',
    TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, instruction, example, is_active)
VALUES (302, 'ApplicationService_relationships', 'relationship', 1, 2, TRUE, TRUE,
    'Derive relationships FROM ApplicationService elements. Valid relationships: Aggregation (to other services or components), Realization (to other behavior/passive elements), Serving (to other structure/behavior elements), Access (to passive elements like DataObject), Flow (to other behavior elements).',
    '{"relationships":[{"source":"as:auth-service","target":"as:user-service","relationship_type":"Serving","confidence":0.8},{"source":"as:auth-service","target":"do:token","relationship_type":"Access","confidence":0.75}]}',
    TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, instruction, example, is_active)
VALUES (303, 'ApplicationInterface_relationships', 'relationship', 1, 3, TRUE, TRUE,
    'Derive relationships FROM ApplicationInterface elements. Valid relationships: Composition (to other structure elements), Aggregation (to structure/behavior/passive), Assignment (to behavior elements), Serving (to other elements), Access (to passive elements).',
    '{"relationships":[{"source":"if:rest-api","target":"as:user-service","relationship_type":"Assignment","confidence":0.85},{"source":"if:rest-api","target":"ac:gateway","relationship_type":"Serving","confidence":0.8}]}',
    TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, instruction, example, is_active)
VALUES (304, 'DataObject_relationships', 'relationship', 1, 4, TRUE, TRUE,
    'Derive relationships FROM DataObject elements. DataObject is a passive element with limited outgoing relationships. Valid relationships: Aggregation (to other data objects or elements), Realization (to other passive elements like BusinessObject).',
    '{"relationships":[{"source":"do:user-profile","target":"bo:customer","relationship_type":"Realization","confidence":0.7},{"source":"do:order","target":"do:order-item","relationship_type":"Aggregation","confidence":0.8}]}',
    TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, instruction, example, is_active)
VALUES (306, 'BusinessObject_relationships', 'relationship', 1, 6, TRUE, TRUE,
    'Derive relationships FROM BusinessObject elements. BusinessObject is a passive element with limited outgoing relationships. Valid relationships: Aggregation (to other objects), Realization (to other passive elements).',
    '{"relationships":[{"source":"bo:customer","target":"bo:address","relationship_type":"Aggregation","confidence":0.8}]}',
    TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, instruction, example, is_active)
VALUES (310, 'TechnologyService_relationships', 'relationship', 1, 10, TRUE, TRUE,
    'Derive relationships FROM TechnologyService elements. Valid relationships: Aggregation (to other elements), Realization (to behavior/passive elements), Serving (to other structure/behavior elements), Access (to passive elements), Flow (to other behavior elements).',
    '{"relationships":[{"source":"tech:postgres","target":"do:user-data","relationship_type":"Access","confidence":0.85},{"source":"tech:postgres","target":"ac:user-service","relationship_type":"Serving","confidence":0.8}]}',
    TRUE);

INSERT INTO derivation_config (id, step_name, phase, version, sequence, enabled, llm, instruction, example, is_active)
VALUES (313, 'SystemSoftware_relationships', 'relationship', 1, 13, TRUE, TRUE,
    'Derive relationships FROM SystemSoftware elements. Valid relationships: Composition (to other structure elements), Aggregation (to structure/behavior/passive), Assignment (to behavior elements), Serving (to other elements), Access (to passive elements).',
    '{"relationships":[{"source":"sys:nodejs","target":"ac:api-server","relationship_type":"Serving","confidence":0.9},{"source":"sys:docker","target":"sys:nodejs","relationship_type":"Composition","confidence":0.85}]}',
    TRUE);

-- =============================================================================
-- DERIVATION PATTERNS: Configurable pattern matching for element derivation
-- =============================================================================

-- ApplicationService patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(1, 'ApplicationService', 'include', 'http_methods', '["get", "post", "put", "patch", "delete"]'),
(2, 'ApplicationService', 'include', 'routing', '["route", "endpoint", "api", "rest"]'),
(3, 'ApplicationService', 'include', 'handlers', '["view", "handler", "controller", "index", "list", "detail", "show"]'),
(4, 'ApplicationService', 'include', 'auth', '["login", "logout", "register", "authenticate"]'),
(5, 'ApplicationService', 'include', 'operations', '["search", "filter", "export", "download", "upload", "import"]'),
(6, 'ApplicationService', 'exclude', 'private', '["_", "private", "internal", "helper"]'),
(7, 'ApplicationService', 'exclude', 'lifecycle', '["__init__", "__del__", "setup", "teardown"]'),
(8, 'ApplicationService', 'exclude', 'utility', '["validate", "parse", "format", "convert", "serialize", "deserialize"]');

-- BusinessObject patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(10, 'BusinessObject', 'exclude', 'base_classes', '["base", "abstract", "mixin", "interface", "protocol"]'),
(11, 'BusinessObject', 'exclude', 'utility', '["helper", "utils", "util", "tools", "common"]'),
(12, 'BusinessObject', 'exclude', 'framework', '["handler", "middleware", "decorator", "wrapper", "factory", "builder", "adapter", "proxy"]'),
(13, 'BusinessObject', 'exclude', 'testing', '["test", "mock", "stub", "fake", "fixture"]'),
(14, 'BusinessObject', 'exclude', 'config', '["config", "settings", "options", "params"]'),
(15, 'BusinessObject', 'exclude', 'errors', '["error", "exception"]'),
(16, 'BusinessObject', 'include', 'people', '["user", "account", "customer", "client", "member"]'),
(17, 'BusinessObject', 'include', 'commerce', '["order", "invoice", "payment", "transaction", "receipt"]'),
(18, 'BusinessObject', 'include', 'inventory', '["product", "item", "catalog", "inventory", "stock"]'),
(19, 'BusinessObject', 'include', 'documents', '["document", "report", "contract", "agreement"]'),
(20, 'BusinessObject', 'include', 'communication', '["message", "notification", "email", "alert"]'),
(21, 'BusinessObject', 'include', 'workflow', '["project", "task", "workflow", "process"]'),
(22, 'BusinessObject', 'include', 'organization', '["employee", "department", "organization", "company"]'),
(23, 'BusinessObject', 'include', 'contact', '["address", "contact", "profile", "preference"]'),
(24, 'BusinessObject', 'include', 'subscription', '["subscription", "plan", "license", "quota"]'),
(25, 'BusinessObject', 'include', 'records', '["position", "entry", "record", "detail"]');

-- BusinessProcess patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(30, 'BusinessProcess', 'exclude', 'lifecycle', '["__init__", "__del__", "__enter__", "__exit__", "__str__", "__repr__", "__eq__", "__hash__"]'),
(31, 'BusinessProcess', 'exclude', 'utility', '["helper", "util", "validate", "parse", "format", "convert", "transform", "serialize", "deserialize"]'),
(32, 'BusinessProcess', 'exclude', 'accessors', '["get_", "set_", "is_", "has_", "_get", "_set"]'),
(33, 'BusinessProcess', 'exclude', 'framework', '["setup", "teardown", "configure", "initialize"]'),
(34, 'BusinessProcess', 'include', 'crud_create', '["create", "add", "insert", "new"]'),
(35, 'BusinessProcess', 'include', 'crud_update', '["update", "modify", "edit", "change"]'),
(36, 'BusinessProcess', 'include', 'crud_delete', '["delete", "remove", "cancel"]'),
(37, 'BusinessProcess', 'include', 'workflow', '["submit", "approve", "reject", "review"]'),
(38, 'BusinessProcess', 'include', 'actions', '["process", "handle", "execute", "run"]'),
(39, 'BusinessProcess', 'include', 'compute', '["generate", "calculate", "compute"]'),
(40, 'BusinessProcess', 'include', 'communication', '["send", "notify", "email", "alert"]'),
(41, 'BusinessProcess', 'include', 'data_transfer', '["export", "import", "sync"]'),
(42, 'BusinessProcess', 'include', 'auth', '["register", "login", "logout", "authenticate"]'),
(43, 'BusinessProcess', 'include', 'commerce', '["checkout", "payment", "order", "invoice", "ship", "deliver", "fulfill"]');

-- BusinessActor patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(50, 'BusinessActor', 'include', 'roles', '["user", "admin", "administrator", "manager", "operator"]'),
(51, 'BusinessActor', 'include', 'external', '["customer", "client", "buyer", "seller", "vendor"]'),
(52, 'BusinessActor', 'include', 'internal', '["employee", "staff", "worker", "agent", "representative"]'),
(53, 'BusinessActor', 'include', 'members', '["member", "subscriber", "owner", "author", "creator"]'),
(54, 'BusinessActor', 'include', 'organizational', '["department", "team", "group", "organization", "company", "partner", "supplier", "provider"]'),
(55, 'BusinessActor', 'include', 'system', '["system", "service", "bot", "scheduler", "daemon"]'),
(56, 'BusinessActor', 'include', 'auth', '["principal", "identity", "account", "role", "permission"]'),
(57, 'BusinessActor', 'exclude', 'data', '["data", "model", "entity", "record", "item", "entry", "request", "response", "message", "event", "log"]'),
(58, 'BusinessActor', 'exclude', 'technical', '["handler", "controller", "service", "repository", "factory", "helper", "util", "config", "settings", "option"]'),
(59, 'BusinessActor', 'exclude', 'errors', '["exception", "error", "validator", "parser"]'),
(60, 'BusinessActor', 'exclude', 'base_classes', '["base", "abstract", "interface", "mixin"]');

-- DataObject patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(70, 'DataObject', 'include', 'database', '["database", "db", "sqlite", "sql"]'),
(71, 'DataObject', 'include', 'config', '["config", "json", "yaml", "yml", "toml", "ini", "env"]'),
(72, 'DataObject', 'include', 'schema', '["schema", "xsd", "dtd"]'),
(73, 'DataObject', 'include', 'data', '["csv", "xml", "data"]'),
(74, 'DataObject', 'exclude', 'source', '["source", "python", "javascript", "typescript"]'),
(75, 'DataObject', 'exclude', 'templates', '["template", "html", "css"]'),
(76, 'DataObject', 'exclude', 'testing', '["test", "spec"]'),
(77, 'DataObject', 'exclude', 'docs', '["docs", "markdown", "readme"]'),
(78, 'DataObject', 'exclude', 'assets', '["asset", "image", "font"]');

-- TechnologyService patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(80, 'TechnologyService', 'include', 'databases', '["sql", "postgres", "mysql", "sqlite", "mongo", "redis", "elastic", "database", "db", "orm", "sqlalchemy", "prisma", "duckdb"]'),
(81, 'TechnologyService', 'include', 'messaging', '["kafka", "rabbitmq", "celery", "amqp", "queue", "pubsub"]'),
(82, 'TechnologyService', 'include', 'http', '["http", "request", "api", "rest", "graphql", "grpc", "websocket", "flask", "fastapi", "django", "express", "axios", "fetch"]'),
(83, 'TechnologyService', 'include', 'cloud', '["aws", "azure", "gcp", "s3", "lambda", "dynamodb", "cloudwatch"]'),
(84, 'TechnologyService', 'include', 'auth', '["oauth", "jwt", "auth", "ldap", "saml"]'),
(85, 'TechnologyService', 'include', 'storage', '["storage", "blob", "file", "minio"]'),
(86, 'TechnologyService', 'include', 'infrastructure', '["docker", "kubernetes", "nginx", "vault", "consul"]'),
(87, 'TechnologyService', 'exclude', 'stdlib', '["os", "sys", "json", "re", "datetime", "time", "logging", "typing", "collections", "functools", "itertools", "pathlib", "io", "copy"]'),
(88, 'TechnologyService', 'exclude', 'stdlib_advanced', '["dataclasses", "enum", "abc", "contextlib", "warnings", "math"]'),
(89, 'TechnologyService', 'exclude', 'testing', '["pytest", "unittest", "mock", "typing_extensions", "pydantic"]'),
(90, 'TechnologyService', 'exclude', 'dev_tools', '["setuptools", "pip", "wheel", "black", "ruff", "mypy", "isort"]');

-- ApplicationInterface patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(91, 'ApplicationInterface', 'include', 'api', '["api", "endpoint", "route", "handler", "controller", "rest", "graphql", "grpc", "webhook", "websocket"]'),
(92, 'ApplicationInterface', 'exclude', 'internal', '["_", "private", "internal"]'),
(93, 'ApplicationInterface', 'exclude', 'utility', '["helper", "util"]');

-- BusinessEvent patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(94, 'BusinessEvent', 'include', 'handler', '["on_", "handle", "emit", "trigger", "dispatch"]'),
(95, 'BusinessEvent', 'include', 'event', '["event", "signal", "message", "notification"]'),
(96, 'BusinessEvent', 'include', 'lifecycle', '["created", "updated", "deleted", "changed"]'),
(97, 'BusinessEvent', 'exclude', 'technical', '["click", "mouse", "key", "scroll"]');

-- BusinessFunction patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(98, 'BusinessFunction', 'include', 'domain', '["service", "domain", "business", "core"]'),
(99, 'BusinessFunction', 'include', 'capability', '["payment", "order", "inventory", "shipping", "billing", "auth"]'),
(100, 'BusinessFunction', 'exclude', 'infrastructure', '["util", "helper", "common", "shared", "lib"]');

-- Device patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(101, 'Device', 'include', 'hardware', '["server", "host", "machine", "hardware", "physical"]'),
(102, 'Device', 'include', 'infra', '["terraform", "cloudformation", "ansible", "infrastructure"]'),
(103, 'Device', 'include', 'storage', '["storage", "disk"]'),
(104, 'Device', 'exclude', 'software', '[".py", ".js", ".ts", "test"]');

-- Node patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(105, 'Node', 'include', 'container', '["docker", "container", "pod"]'),
(106, 'Node', 'include', 'k8s', '["kubernetes", "deployment", "k8s", "helm"]'),
(107, 'Node', 'include', 'cloud', '["ec2", "instance", "vm", "lambda", "function"]'),
(108, 'Node', 'exclude', 'config', '[".env", "config."]'),
(109, 'Node', 'exclude', 'test', '["test"]');

-- SystemSoftware patterns
INSERT INTO derivation_patterns (id, step_name, pattern_type, pattern_category, patterns) VALUES
(110, 'SystemSoftware', 'include', 'runtime', '["python", "node", "java", "jvm", "runtime"]'),
(111, 'SystemSoftware', 'include', 'database', '["postgres", "mysql", "mongo", "redis"]'),
(112, 'SystemSoftware', 'include', 'messaging', '["kafka", "rabbitmq", "celery"]'),
(113, 'SystemSoftware', 'include', 'webserver', '["nginx", "apache"]'),
(114, 'SystemSoftware', 'include', 'container', '["docker"]'),
(115, 'SystemSoftware', 'exclude', 'library', '["utils", "helper", "typing"]');
