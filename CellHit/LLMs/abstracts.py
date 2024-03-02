from Bio import Entrez

def fetch_abstracts(term, k,mail=None):
    # Configure the Entrez API
    Entrez.email = mail
    
    # Step 1: Search for PubMed IDs based on the query term
    handle = Entrez.esearch(db="pubmed", term=term, retmax=k, sort="relevance")
    search_results = Entrez.read(handle)
    handle.close()
    id_list = search_results["IdList"]
    
    # Step 2: Fetch details for each PubMed ID
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml")
    papers = Entrez.read(handle)['PubmedArticle']
    handle.close()
    
    # Step 3: Parse and print the abstracts
    abstracts = []
    for i, paper in enumerate(papers):
        try:
            abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
            abstracts.append(abstract)
        except KeyError:
            print(f"Abstract {i+1} is not available.\n")
    
    return abstracts