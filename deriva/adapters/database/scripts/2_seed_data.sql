-- Seed Data: File Type Registry with Chunking Configuration
-- Total entries: 66

-- =============================================================================
-- ASSET FILES
-- =============================================================================
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.jpg', 'asset', 'image');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.png', 'asset', 'image');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.svg', 'asset', 'image');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.ttf', 'asset', 'font');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.woff', 'asset', 'font');

-- =============================================================================
-- BUILD FILES
-- =============================================================================
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('dockerfile', 'build', 'docker');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('makefile', 'build', 'make');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('pom.xml', 'build', 'maven');

-- =============================================================================
-- CONFIG FILES (with chunking config where applicable)
-- =============================================================================
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.env', 'config', 'env');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.ini', 'config', 'ini');
INSERT INTO file_type_registry (extension, file_type, subtype, chunk_delimiter, chunk_overlap) VALUES ('.json', 'config', 'json', NULL, 0);
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.toml', 'config', 'toml');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.xml', 'config', 'xml');
INSERT INTO file_type_registry (extension, file_type, subtype, chunk_delimiter, chunk_overlap) VALUES ('.yaml', 'config', 'yaml', NULL, 0);
INSERT INTO file_type_registry (extension, file_type, subtype, chunk_delimiter, chunk_overlap) VALUES ('.yml', 'config', 'yaml', NULL, 0);

-- =============================================================================
-- DATA FILES
-- =============================================================================
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.csv', 'data', 'csv');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.parquet', 'data', 'parquet');

-- =============================================================================
-- DEPENDENCY FILES (package/dependency manifest files - matched by full filename)
-- =============================================================================
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('requirements.txt', 'dependency', 'python');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('setup.py', 'dependency', 'python');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('setup.cfg', 'dependency', 'python');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('pyproject.toml', 'dependency', 'python');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('package.json', 'dependency', 'javascript');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('package-lock.json', 'dependency', 'javascript');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('yarn.lock', 'dependency', 'javascript');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('go.mod', 'dependency', 'golang');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('go.sum', 'dependency', 'golang');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('Cargo.toml', 'dependency', 'rust');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('Cargo.lock', 'dependency', 'rust');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('Gemfile', 'dependency', 'ruby');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('Gemfile.lock', 'dependency', 'ruby');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('composer.json', 'dependency', 'php');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('composer.lock', 'dependency', 'php');

-- =============================================================================
-- DOCS FILES (with chunking config for markdown)
-- =============================================================================
INSERT INTO file_type_registry (extension, file_type, subtype, chunk_delimiter, chunk_overlap) VALUES ('.md', 'docs', 'markdown', '
## ', 0);
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.rst', 'docs', 'restructuredtext');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.txt', 'docs', 'text');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.adoc', 'docs', 'asciidoc');
INSERT INTO file_type_registry (extension, file_type, subtype, chunk_delimiter, chunk_overlap) VALUES ('.markdown', 'docs', 'markdown', '
## ', 0);

-- =============================================================================
-- EXCLUDE PATTERNS
-- =============================================================================
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.git', 'exclude', 'vcs');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.venv', 'exclude', 'virtualenv');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('__pycache__', 'exclude', 'cache');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('build', 'exclude', 'build');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('dist', 'exclude', 'build');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('node_modules', 'exclude', 'dependencies');

-- =============================================================================
-- SOURCE FILES (with chunking config for code files)
-- =============================================================================
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.c', 'source', 'c');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.cpp', 'source', 'cpp');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.cs', 'source', 'csharp');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.css', 'source', 'stylesheet');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.go', 'source', 'golang');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.html', 'source', 'html');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.java', 'source', 'java');
INSERT INTO file_type_registry (extension, file_type, subtype, chunk_delimiter, chunk_overlap) VALUES ('.js', 'source', 'javascript', '
function ', 0);
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.php', 'source', 'php');
INSERT INTO file_type_registry (extension, file_type, subtype, chunk_delimiter, chunk_overlap) VALUES ('.py', 'source', 'python', '
class ', 0);
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.rb', 'source', 'ruby');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.rs', 'source', 'rust');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.scss', 'source', 'stylesheet');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('.sh', 'source', 'shell');
INSERT INTO file_type_registry (extension, file_type, subtype, chunk_delimiter, chunk_overlap) VALUES ('.sql', 'source', 'sql', ';
', 0);
INSERT INTO file_type_registry (extension, file_type, subtype, chunk_delimiter, chunk_overlap) VALUES ('.ts', 'source', 'typescript', '
function ', 0);

-- =============================================================================
-- TEST FILES (wildcard patterns - matched using fnmatch)
-- =============================================================================
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('*.spec.js', 'test', 'javascript');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('*.test.js', 'test', 'javascript');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('*_test.py', 'test', 'python');
INSERT INTO file_type_registry (extension, file_type, subtype) VALUES ('test_*.py', 'test', 'python');
