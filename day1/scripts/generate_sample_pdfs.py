"""
Generate sample PDF documents for RAG training
This script creates 3 sample business PDFs with realistic content
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import os

def create_company_handbook():
    """Create a sample company handbook PDF"""
    filename = "docs/company_handbook.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='darkblue',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("TechCorp Inc. Employee Handbook", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Content
    content = [
        ("Company Overview", """
        TechCorp Inc. is a leading technology solutions provider established in 2015. 
        We specialize in artificial intelligence, machine learning, and cloud computing services. 
        Our mission is to empower businesses through innovative technology solutions that drive 
        digital transformation and operational excellence.
        """),
        
        ("Working Hours and Remote Policy", """
        Standard working hours are 9:00 AM to 6:00 PM, Monday through Friday. We offer a 
        flexible hybrid work policy allowing employees to work remotely up to 3 days per week. 
        Core hours (10:00 AM - 3:00 PM) require all team members to be available for 
        collaboration and meetings.
        """),
        
        ("Leave Policy", """
        Full-time employees are entitled to 20 days of paid annual leave, 10 days of sick leave, 
        and 12 public holidays per year. Leave requests should be submitted through the HR portal 
        at least 2 weeks in advance for planned vacations. Unused leave can be carried forward 
        up to a maximum of 5 days to the following year.
        """),
        
        ("Benefits Package", """
        TechCorp offers comprehensive benefits including health insurance coverage for employees 
        and dependents, dental and vision care, life insurance, and a 401(k) retirement plan 
        with up to 5% company matching. Additional perks include gym membership reimbursement, 
        professional development budget of $2,000 annually, and stock options for senior positions.
        """),
        
        ("Performance Reviews", """
        Performance reviews are conducted bi-annually in January and July. Employees meet with 
        their managers to discuss achievements, areas for improvement, and career development goals. 
        Reviews directly impact annual compensation adjustments and promotion decisions. 
        Self-assessments are required one week before the scheduled review meeting.
        """),
        
        ("Code of Conduct", """
        All employees are expected to maintain professional behavior, respect diversity and inclusion, 
        protect confidential information, and comply with company policies. Harassment, discrimination, 
        or unethical behavior will not be tolerated. Report any concerns to HR immediately through 
        the anonymous hotline or hr@techcorp.com.
        """),
    ]
    
    for heading, text in content:
        story.append(Paragraph(heading, styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(text, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
    
    doc.build(story)
    print(f"✅ Created: {filename}")

def create_product_guide():
    """Create a sample product guide PDF"""
    filename = "docs/product_guide.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='darkgreen',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("CloudMaster Pro - Product Guide", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    content = [
        ("Product Overview", """
        CloudMaster Pro is an enterprise-grade cloud management platform designed to simplify 
        multi-cloud operations. It provides unified visibility, cost optimization, security 
        compliance, and automated workflows across AWS, Azure, and Google Cloud Platform. 
        Version 3.5 introduces AI-powered recommendations and enhanced security features.
        """),
        
        ("Key Features", """
        <b>1. Multi-Cloud Dashboard:</b> Centralized view of all cloud resources across providers 
        with real-time metrics and alerts.<br/>
        <b>2. Cost Optimization:</b> AI-driven insights to reduce cloud spending by up to 40% through 
        rightsizing recommendations and waste elimination.<br/>
        <b>3. Security Compliance:</b> Automated compliance checks for SOC2, HIPAA, GDPR with 
        one-click remediation.<br/>
        <b>4. Workflow Automation:</b> Pre-built templates for common tasks like backup, 
        disaster recovery, and resource provisioning.
        """),
        
        ("Getting Started", """
        To begin using CloudMaster Pro, log in to your account at portal.cloudmaster.com. 
        Navigate to Settings > Cloud Connections and add your cloud provider credentials. 
        The platform uses read-only IAM roles for security. Once connected, the initial 
        discovery process takes 5-10 minutes to scan all resources. You'll receive an email 
        when the scan completes.
        """),
        
        ("Dashboard Navigation", """
        The main dashboard displays four primary widgets: Cost Summary, Resource Inventory, 
        Security Alerts, and Performance Metrics. Click any widget to drill down into detailed 
        views. Use the filter bar at the top to narrow results by cloud provider, region, 
        or time period. Customize your dashboard by dragging widgets or adding new ones from 
        the Widget Library.
        """),
        
        ("Cost Optimization Tools", """
        Access cost optimization recommendations from the left sidebar menu. The system analyzes 
        your usage patterns and identifies idle resources, oversized instances, and unused storage. 
        Each recommendation includes estimated monthly savings and implementation difficulty. 
        Apply recommendations with one click or schedule them for maintenance windows. Track 
        savings over time in the ROI Report section.
        """),
        
        ("Security and Compliance", """
        The Security Center provides a compliance score based on your selected frameworks. 
        Red indicators show critical violations requiring immediate attention. Yellow warnings 
        indicate best practice deviations. Review detailed findings and click "Remediate" to 
        automatically fix issues where possible. Schedule weekly compliance reports to be sent 
        to your email or Slack channel.
        """),
        
        ("Automation Workflows", """
        Create custom automation workflows using our drag-and-drop editor. Start with a trigger 
        (schedule, event, or webhook), add action steps (provision, backup, scale), and define 
        conditions. Test workflows in sandbox mode before enabling for production. Monitor 
        workflow execution history and set up failure notifications. Pre-built templates are 
        available for common scenarios.
        """),
        
        ("Support and Training", """
        Access documentation at docs.cloudmaster.com or contact support@cloudmaster.com for 
        assistance. Live chat support is available 24/7 for Enterprise customers. Free training 
        webinars are held monthly covering basic to advanced topics. Schedule private training 
        sessions for your team through the customer portal. Join our community forum to share 
        best practices with other users.
        """),
    ]
    
    for heading, text in content:
        story.append(Paragraph(heading, styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(text, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
    
    doc.build(story)
    print(f"✅ Created: {filename}")

def create_technical_manual():
    """Create a sample technical manual PDF"""
    filename = "docs/technical_manual.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='darkred',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("DataFlow API - Technical Reference", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    content = [
        ("Introduction", """
        The DataFlow API is a RESTful web service that enables real-time data integration 
        and transformation. It supports JSON and XML formats with webhook notifications for 
        asynchronous processing. This manual covers API v2.0 released in October 2024. 
        Base URL: https://api.dataflow.io/v2
        """),
        
        ("Authentication", """
        All API requests require authentication using API keys. Include your key in the 
        Authorization header: <i>Authorization: Bearer YOUR_API_KEY</i>. Generate API keys 
        from the dashboard under Settings > API Keys. Keys can have read-only or read-write 
        permissions. Rotate keys every 90 days for security. Rate limits: 1000 requests per 
        hour for standard plans, 10000 for enterprise.
        """),
        
        ("Endpoints - Data Ingestion", """
        <b>POST /api/v2/ingest</b><br/>
        Upload data for processing. Accepts JSON payload up to 10MB.<br/>
        Request body: {\"source\": \"string\", \"data\": \"object\", \"transformations\": \"array\"}<br/>
        Response: {\"job_id\": \"string\", \"status\": \"queued\", \"eta_seconds\": integer}<br/>
        Example: POST https://api.dataflow.io/v2/ingest with Content-Type: application/json
        """),
        
        ("Endpoints - Job Status", """
        <b>GET /api/v2/jobs/{job_id}</b><br/>
        Check processing status of a submitted job.<br/>
        Path parameter: job_id (string, required)<br/>
        Response: {\"job_id\": \"string\", \"status\": \"processing|completed|failed\", 
        \"progress\": integer, \"result_url\": \"string\"}<br/>
        Status values: queued, processing, completed, failed. Poll every 5-10 seconds.
        """),
        
        ("Endpoints - Data Retrieval", """
        <b>GET /api/v2/data/{dataset_id}</b><br/>
        Retrieve processed data by dataset ID.<br/>
        Query parameters: format (json|xml|csv), limit (integer, max 1000), offset (integer)<br/>
        Response: {\"dataset_id\": \"string\", \"records\": array, \"total_count\": integer, 
        \"next_page\": \"string\"}<br/>
        Use pagination for large datasets. Cache responses for 5 minutes.
        """),
        
        ("Data Transformations", """
        The API supports built-in transformations applied during ingestion. Available 
        transformations: filter (remove records by condition), map (transform field values), 
        aggregate (group and summarize), join (merge with reference data), validate (schema 
        validation). Specify transformations as an array in the ingest payload. Transformations 
        are applied in order. Complex transformations may increase processing time.
        """),
        
        ("Error Handling", """
        API returns standard HTTP status codes. 200: Success, 400: Bad Request (validation error), 
        401: Unauthorized (invalid API key), 429: Rate Limit Exceeded, 500: Internal Server Error. 
        Error responses include: {\"error\": \"string\", \"message\": \"string\", \"code\": \"string\"}. 
        Retry failed requests with exponential backoff. Contact support if errors persist for 500 status codes.
        """),
        
        ("Webhooks", """
        Configure webhooks to receive notifications when jobs complete. Set webhook URL in 
        dashboard under Settings > Webhooks. Webhook payload: {\"event\": \"job.completed\", 
        \"job_id\": \"string\", \"status\": \"string\", \"timestamp\": \"ISO8601\"}. 
        Webhooks include HMAC signature in X-Signature header for verification. Respond with 
        200 status within 5 seconds. Failed webhook deliveries are retried 3 times.
        """),
        
        ("Best Practices", """
        1. Batch multiple records in a single ingest request to reduce API calls.<br/>
        2. Use webhooks instead of polling for job status when possible.<br/>
        3. Implement exponential backoff for rate limit errors.<br/>
        4. Cache GET responses to minimize redundant requests.<br/>
        5. Validate data client-side before sending to reduce processing failures.<br/>
        6. Use compression (gzip) for large payloads to improve transfer speed.<br/>
        7. Monitor your API usage in the dashboard to avoid unexpected overages.
        """),
        
        ("SDK Libraries", """
        Official SDK libraries are available for Python, JavaScript, Java, and Go. Install via 
        package managers: pip install dataflow-sdk, npm install @dataflow/sdk. SDKs handle 
        authentication, retries, and pagination automatically. Refer to SDK documentation at 
        docs.dataflow.io/sdks. Community libraries exist for Ruby, PHP, and .NET. Example code 
        snippets are provided in the documentation for common use cases.
        """),
    ]
    
    for heading, text in content:
        story.append(Paragraph(heading, styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(text, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
    
    doc.build(story)
    print(f"✅ Created: {filename}")

if __name__ == "__main__":
    # Create docs directory if it doesn't exist
    os.makedirs("docs", exist_ok=True)
    
    print("📄 Generating sample PDF documents for training...")
    print()
    
    create_company_handbook()
    create_product_guide()
    create_technical_manual()
    
    print()
    print("✅ All sample PDFs created successfully!")
    print("📁 Files are located in the 'docs/' folder")
