"""
 parsers.py: This file will handle the parsing of the raw text extracted using loaders.py
"""
def clean_metadata(document):
    
    if 'type' not in document.metadata or document.metadata['type'] is None:
        document.metadata['type'] = 'text'
    
    return document

def clean_document(document):
    # TODO: Add more cleaning logic
    document.page_content = document.page_content.strip()
    return document

def parse_document(document):
    # Clean up metadata
    document = clean_metadata(document)
    # Clean up document
    document = clean_document(document)
    return document

def parse_dir(docs):
    # Parse all the documents
    parsed_documents = []
    for doc in docs:
        parsed_document = parse_document(doc)
        parsed_documents.append(parsed_document)
    
    return parsed_documents
