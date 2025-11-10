import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import time
import logging

logger = logging.getLogger(__name__)

def search_mesh_term(condition, delay=0.35, filter_diseases_only=True):
    """
    Search for a MeSH term using NCBI E-utilities API.
    
    Args:
        condition: The condition string to search for
        delay: Delay between API calls to respect rate limits (default 0.35s for 3 req/sec)
        filter_diseases_only: If True, only return terms in Disease category (tree numbers starting with 'C')
    
    Returns:
        Dictionary with mesh_term, mesh_id, tree_numbers, categories, and is_disease if found, None otherwise
    """
    try:
        # URL encode the search term
        search_term = urllib.parse.quote(condition)
        
        # Step 1: Search for MeSH term using esearch
        esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh&term={search_term}&retmode=xml&retmax=1&sort=relevance"
        
        with urllib.request.urlopen(esearch_url) as response:
            search_xml = response.read()
        
        # Parse search results
        search_root = ET.fromstring(search_xml)
        id_list = search_root.find('.//IdList')
        
        if id_list is None or len(id_list) == 0:
            logger.info(f"No MeSH match found for: {condition}")
            return None
        
        # Get the first (best) match ID
        mesh_id = id_list[0].text
        
        # Add delay before next request
        time.sleep(delay)
        
        # Step 2: Get full MeSH term details using esummary
        esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=mesh&id={mesh_id}&retmode=xml"
        
        with urllib.request.urlopen(esummary_url) as response:
            summary_xml = response.read()
        
        # Parse summary to get MeSH descriptor name and tree numbers
        summary_root = ET.fromstring(summary_xml)
        
        # Get MeSH descriptor name (first Item in DS_MeshTerms)
        mesh_terms_list = summary_root.find('.//Item[@Name="DS_MeshTerms"]')
        
        if mesh_terms_list is None or len(list(mesh_terms_list)) == 0:
            logger.info(f"No MeSH descriptor found for ID: {mesh_id}")
            return None
        
        mesh_term = list(mesh_terms_list)[0].text
        
        # Extract tree numbers from DS_IdxLinks
        tree_numbers = []
        idx_links = summary_root.find('.//Item[@Name="DS_IdxLinks"]')
        
        if idx_links is not None:
            for item in idx_links.findall('.//Item[@Name="TreeNum"]'):
                if item.text:
                    tree_numbers.append(item.text)
        
        # Determine categories from tree numbers (first character indicates category)
        categories = set()
        for tn in tree_numbers:
            if tn and tn[0].isalpha():
                categories.add(tn[0])
        
        # Check if this is a disease (category C)
        is_disease = any((tn.startswith('C') or tn.startswith('F')) for tn in tree_numbers if tn)
        
        # Apply disease filter if requested
        if filter_diseases_only and not is_disease:
            logger.info(f"Filtered out non-disease term: '{mesh_term}' (categories: {', '.join(sorted(categories))})")
            return None
        
        result = {
            'mesh_term': mesh_term,
            'mesh_id': mesh_id,
            'tree_numbers': tree_numbers,
            'categories': sorted(list(categories)),
            'is_disease': is_disease
        }
        
        category_str = ', '.join(sorted(categories)) if categories else 'None'
        logger.info(f"Matched '{condition}' -> '{mesh_term}' ({mesh_id}) [Categories: {category_str}, Disease: {is_disease}]")
        return result
            
    except Exception as e:
        logger.error(f"Error searching MeSH for '{condition}': {str(e)}")
        return None