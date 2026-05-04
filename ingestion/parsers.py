"""
 parsers.py: This file will handle the parsing of the raw text extracted using loaders.py
"""
def clean_metadata(document):
    
    if 'type' not in document.metadata or document.metadata['type'] is None:
        document.metadata['type'] = 'text'
    elif document.metadata['type'] == 'code':
        # Safely check for extension
        extension = document.metadata.get('extension', '')
        if extension == '.py':
            document.metadata['language'] = 'python'
        elif extension == '.js':
            document.metadata['language'] = 'javascript'
        elif extension == '.html':
            document.metadata['language'] = 'html'
        elif extension == '.css':
            document.metadata['language'] = 'css'
        else:
            document.metadata['language'] = 'Unknown'
    
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
