# Integration Tests

Comprehensive end-to-end testing for VEDA API.

## Running Tests

### All Integration Tests
```bash
pytest tests/integration/ -v
```

### Specific Test Classes
```bash
# Authentication tests
pytest tests/integration/test_e2e_workflow.py::TestAuthentication -v

# Workflow creation
pytest tests/integration/test_e2e_workflow.py::TestWorkflowCreation -v

# Workflow completion (long-running)
pytest tests/integration/test_e2e_workflow.py::TestWorkflowPolling -v -s
```

### With Coverage
```bash
pytest tests/integration/ --cov=veda --cov-report=html
```

## Test Categories

1. **Authentication Tests** - Login, token validation
2. **Workflow Creation Tests** - Create workflows with different datasets
3. **Workflow Retrieval Tests** - Get status, list workflows
4. **Workflow Polling Tests** - Wait for completion
5. **Report Generation Tests** - HTML report generation
6. **System Endpoint Tests** - Health, stats, root
7. **Error Handling Tests** - Invalid requests, auth failures
8. **Concurrent Operation Tests** - Multiple simultaneous requests

## Prerequisites

- API server running on `http://localhost:8000`
- Example datasets in `examples/datasets/`
- Valid credentials (admin/admin123)

## Expected Results

All tests should pass with:
- ✅ Authentication working
- ✅ Workflows created successfully
- ✅ Status endpoints responding
- ✅ Proper error handling