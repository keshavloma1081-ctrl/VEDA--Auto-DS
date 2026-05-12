"""
Test Data Sources Agents (Agents 11-15)
"""
from veda.agents.data_sources.pdf_processor import PDFProcessorAgent
from veda.agents.data_sources.sql_database import SQLDatabaseAgent
from veda.agents.data_sources.rest_api import RESTAPIAgent
from veda.agents.data_sources.excel_json_parser import ExcelJSONParserAgent
from veda.agents.data_sources.cloud_storage import CloudStorageAgent
import json

def test_data_sources():
    print("\n" + "="*60)
    print("TESTING DATA SOURCES AGENTS (11-15)")
    print("="*60)
    
    # Agent 11: PDF Processor
    print("\n[1/5] PDF Processor Agent...")
    pdf_agent = PDFProcessorAgent()
    pdf_state = {
        "pdf_path": "test_data/data_sources/pdfs/quarterly_report.txt",
        "extraction_task": "Extract sales data and regional performance table"
    }
    pdf_result = pdf_agent.execute(pdf_state)
    print(f"Result: {json.dumps(pdf_result, indent=2)}")
    
    # Agent 12: SQL Database
    print("\n[2/5] SQL Database Agent...")
    sql_agent = SQLDatabaseAgent()
    sql_state = {
        "schema_path": "test_data/data_sources/databases/ecommerce_schema.sql",
        "user_query": "Get top 10 customers by total order value in the last 30 days"
    }
    sql_result = sql_agent.execute(sql_state)
    print(f"Result: {json.dumps(sql_result, indent=2)}")
    
    # Agent 13: REST API
    print("\n[3/5] REST API Agent...")
    api_agent = RESTAPIAgent()
    api_state = {
        "spec_path": "test_data/data_sources/apis/api_spec.json",
        "data_requirements": "Fetch all active users with lifetime value > 2000"
    }
    api_result = api_agent.execute(api_state)
    print(f"Result: {json.dumps(api_result, indent=2)}")
    
    # Agent 14: Excel/JSON Parser
    print("\n[4/5] Excel/JSON Parser Agent...")
    parser_agent = ExcelJSONParserAgent()
    parser_state = {
        "file_type": "csv",
        "file_path": "test_data/data_sources/files/products_inventory.csv"
    }
    parser_result = parser_agent.execute(parser_state)
    print(f"Result: {json.dumps(parser_result, indent=2)}")
    
    # Agent 15: Cloud Storage
    print("\n[5/5] Cloud Storage Agent...")
    cloud_agent = CloudStorageAgent()
    cloud_state = {
        "inventory_path": "test_data/data_sources/cloud/s3_inventory.md",
        "task": "List all CSV files larger than 2MB from the raw data folder"
    }
    cloud_result = cloud_agent.execute(cloud_state)
    print(f"Result: {json.dumps(cloud_result, indent=2)}")
    
    print("\n" + "="*60)
    print("DATA SOURCES DOMAIN COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_data_sources()