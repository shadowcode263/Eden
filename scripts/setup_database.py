import os
import django
from django.conf import settings
from django.core.management import execute_from_command_line

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from django.db import connection

# Create database tables
sql_commands = [
    """
    CREATE TABLE IF NOT EXISTS brain_networks (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        embedding_dim INTEGER NOT NULL DEFAULT 64,
        beta FLOAT NOT NULL DEFAULT 20.0,
        learning_rate FLOAT NOT NULL DEFAULT 0.1,
        merkle_root VARCHAR(64),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS brain_patterns (
        id SERIAL PRIMARY KEY,
        network_id INTEGER REFERENCES brain_networks(id) ON DELETE CASCADE,
        pattern_hash VARCHAR(64) UNIQUE NOT NULL,
        pattern_data BYTEA NOT NULL,
        embedding_data BYTEA NOT NULL,
        usage_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS brain_retrievals (
        id SERIAL PRIMARY KEY,
        network_id INTEGER REFERENCES brain_networks(id) ON DELETE CASCADE,
        query_text TEXT,
        retrieved_pattern_id INTEGER REFERENCES brain_patterns(id),
        confidence_score FLOAT,
        retrieval_steps INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_patterns_network ON brain_patterns(network_id);
    CREATE INDEX IF NOT EXISTS idx_patterns_hash ON brain_patterns(pattern_hash);
    CREATE INDEX IF NOT EXISTS idx_retrievals_network ON brain_retrievals(network_id);
    """
]

with connection.cursor() as cursor:
    for sql in sql_commands:
        cursor.execute(sql)

print("✅ Database tables created successfully!")
