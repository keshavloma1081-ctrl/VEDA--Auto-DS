"""Add performance indexes

Revision ID: 002
Revises: 001
Create Date: 2026-05-14
"""
from alembic import op

revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_index('idx_workflow_created_by', 'workflows', ['created_by'])
    op.create_index('idx_model_workflow', 'models', ['workflow_id'])
    op.create_index('idx_prediction_model_created', 'predictions', ['model_id', 'created_at'])
    op.create_index('idx_monitoring_model', 'model_monitoring_logs', ['model_id'])
    op.create_index('idx_audit_user', 'audit_logs', ['user_id'])
    op.create_index('idx_dataset_workflow', 'dataset_versions', ['workflow_id'])

def downgrade() -> None:
    op.drop_index('idx_workflow_created_by', 'workflows')
    op.drop_index('idx_model_workflow', 'models')
    op.drop_index('idx_prediction_model_created', 'predictions')
    op.drop_index('idx_monitoring_model', 'model_monitoring_logs')
    op.drop_index('idx_audit_user', 'audit_logs')
    op.drop_index('idx_dataset_workflow', 'dataset_versions')