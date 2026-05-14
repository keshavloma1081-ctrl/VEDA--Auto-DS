"""Initial schema - all VEDA tables

Revision ID: 001
Revises:
Create Date: 2026-05-14
"""
from alembic import op
import sqlalchemy as sa

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table('workflows',
        sa.Column('job_id', sa.String(36), primary_key=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='submitted'),
        sa.Column('progress', sa.Float, server_default='0.0'),
        sa.Column('dataset_path', sa.Text, nullable=False),
        sa.Column('goal', sa.Text, nullable=False),
        sa.Column('config', sa.JSON, nullable=True),
        sa.Column('current_step', sa.String(100), nullable=True),
        sa.Column('total_steps', sa.Integer, server_default='11'),
        sa.Column('completed_steps', sa.Integer, server_default='0'),
        sa.Column('result', sa.JSON, nullable=True),
        sa.Column('error', sa.Text, nullable=True),
        sa.Column('error_traceback', sa.Text, nullable=True),
        sa.Column('model_type', sa.String(100), nullable=True),
        sa.Column('accuracy', sa.Float, nullable=True),
        sa.Column('f1_score', sa.Float, nullable=True),
        sa.Column('auc_roc', sa.Float, nullable=True),
        sa.Column('task_type', sa.String(50), nullable=True),
        sa.Column('dataset_rows', sa.Integer, nullable=True),
        sa.Column('dataset_cols', sa.Integer, nullable=True),
        sa.Column('dataset_size_mb', sa.Float, nullable=True),
        sa.Column('model_path', sa.Text, nullable=True),
        sa.Column('report_path', sa.Text, nullable=True),
        sa.Column('mlflow_run_id', sa.String(100), nullable=True),
        sa.Column('created_by', sa.String(100), nullable=True),
        sa.Column('celery_task_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=True),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
    )
    op.create_index('idx_workflow_status', 'workflows', ['status'])
    op.create_index('idx_workflow_created', 'workflows', ['created_at'])
    op.create_index('idx_workflow_status_created', 'workflows', ['status', 'created_at'])

    op.create_table('models',
        sa.Column('model_id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('version', sa.String(50), server_default='1.0.0'),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('model_type', sa.String(50), nullable=True),
        sa.Column('task_type', sa.String(50), nullable=True),
        sa.Column('metrics', sa.JSON, nullable=True),
        sa.Column('hyperparameters', sa.JSON, nullable=True),
        sa.Column('feature_importance', sa.JSON, nullable=True),
        sa.Column('file_path', sa.Text, nullable=False),
        sa.Column('file_size_mb', sa.Float, nullable=True),
        sa.Column('mlflow_run_id', sa.String(100), nullable=True),
        sa.Column('workflow_id', sa.String(36), nullable=True),
        sa.Column('training_duration_seconds', sa.Float, nullable=True),
        sa.Column('is_active', sa.Boolean, server_default='1'),
        sa.Column('stage', sa.String(50), server_default='development'),
        sa.Column('dataset_rows', sa.Integer, nullable=True),
        sa.Column('dataset_cols', sa.Integer, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=True),
    )
    op.create_index('idx_model_name', 'models', ['name'])

    op.create_table('predictions',
        sa.Column('prediction_id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.String(36), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('input_data', sa.JSON, nullable=False),
        sa.Column('input_hash', sa.String(64), nullable=True),
        sa.Column('predictions', sa.JSON, nullable=False),
        sa.Column('probabilities', sa.JSON, nullable=True),
        sa.Column('prediction_label', sa.String(100), nullable=True),
        sa.Column('inference_time_ms', sa.Float, nullable=True),
        sa.Column('actual_label', sa.String(100), nullable=True),
        sa.Column('is_correct', sa.Boolean, nullable=True),
        sa.Column('request_id', sa.String(36), nullable=True),
        sa.Column('client_ip', sa.String(50), nullable=True),
        sa.Column('user_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
    )
    op.create_index('idx_prediction_model', 'predictions', ['model_id'])
    op.create_index('idx_prediction_created', 'predictions', ['created_at'])

    op.create_table('dataset_versions',
        sa.Column('version_id', sa.String(36), primary_key=True),
        sa.Column('file_path', sa.Text, nullable=False),
        sa.Column('file_name', sa.String(500), nullable=True),
        sa.Column('file_hash', sa.String(64), nullable=True),
        sa.Column('file_size_bytes', sa.Integer, nullable=True),
        sa.Column('num_rows', sa.Integer, nullable=True),
        sa.Column('num_cols', sa.Integer, nullable=True),
        sa.Column('num_missing', sa.Integer, nullable=True),
        sa.Column('column_names', sa.JSON, nullable=True),
        sa.Column('dtypes', sa.JSON, nullable=True),
        sa.Column('statistics', sa.JSON, nullable=True),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('workflow_id', sa.String(36), nullable=True),
        sa.Column('created_by', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
    )
    op.create_index('idx_dataset_hash', 'dataset_versions', ['file_hash'])

    op.create_table('model_monitoring_logs',
        sa.Column('log_id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.String(36), nullable=False),
        sa.Column('period_start', sa.DateTime, nullable=False),
        sa.Column('period_end', sa.DateTime, nullable=False),
        sa.Column('total_predictions', sa.Integer, server_default='0'),
        sa.Column('accuracy', sa.Float, nullable=True),
        sa.Column('f1_score', sa.Float, nullable=True),
        sa.Column('feature_drift_scores', sa.JSON, nullable=True),
        sa.Column('prediction_drift_score', sa.Float, nullable=True),
        sa.Column('data_drift_detected', sa.Boolean, server_default='0'),
        sa.Column('concept_drift_detected', sa.Boolean, server_default='0'),
        sa.Column('alert_triggered', sa.Boolean, server_default='0'),
        sa.Column('alert_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
    )

    op.create_table('ab_experiments',
        sa.Column('experiment_id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('model_a_id', sa.String(36), nullable=False),
        sa.Column('model_b_id', sa.String(36), nullable=False),
        sa.Column('traffic_split', sa.Float, server_default='0.5'),
        sa.Column('status', sa.String(50), server_default='running'),
        sa.Column('winner', sa.String(10), nullable=True),
        sa.Column('model_a_requests', sa.Integer, server_default='0'),
        sa.Column('model_b_requests', sa.Integer, server_default='0'),
        sa.Column('model_a_accuracy', sa.Float, nullable=True),
        sa.Column('model_b_accuracy', sa.Float, nullable=True),
        sa.Column('p_value', sa.Float, nullable=True),
        sa.Column('is_significant', sa.Boolean, nullable=True),
        sa.Column('started_at', sa.DateTime, nullable=False),
        sa.Column('ended_at', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
    )

    op.create_table('audit_logs',
        sa.Column('log_id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=True),
        sa.Column('resource_id', sa.String(36), nullable=True),
        sa.Column('user_id', sa.String(100), nullable=True),
        sa.Column('user_role', sa.String(50), nullable=True),
        sa.Column('client_ip', sa.String(50), nullable=True),
        sa.Column('method', sa.String(10), nullable=True),
        sa.Column('endpoint', sa.String(200), nullable=True),
        sa.Column('status_code', sa.Integer, nullable=True),
        sa.Column('duration_ms', sa.Float, nullable=True),
        sa.Column('details', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
    )
    op.create_index('idx_audit_action', 'audit_logs', ['action'])
    op.create_index('idx_audit_created', 'audit_logs', ['created_at'])

def downgrade() -> None:
    op.drop_table('audit_logs')
    op.drop_table('ab_experiments')
    op.drop_table('model_monitoring_logs')
    op.drop_table('dataset_versions')
    op.drop_table('predictions')
    op.drop_table('models')
    op.drop_table('workflows')