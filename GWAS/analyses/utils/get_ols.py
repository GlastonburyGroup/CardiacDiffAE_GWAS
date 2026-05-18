import requests
import json
import pickle
import logging
import argparse
from pandasgwas.get_studies import get_studies

#fix eqs
#rewrite abstract
#complexity
#time for inference - negligable ( scale difference)
#abstract: complex motivate and the contrib
#to the best of our knowledge, add if there is

def get_parent(efo_id, retries=10):
    url = f"http://www.ebi.ac.uk/ols4/api/ontologies/efo/parents?id={efo_id}"
    headers = {
            "Content-Type": "application/json",
        }
    response = requests.post(url, headers=headers, verify=True)

    if response.status_code != 200:
        logging.error(f"Error occurred while fetching the parent: Received status code {response.status_code} instead of 200.")
        if retries > 0:
            return get_parent(efo_id, retries - 1)
        logging.error("Max retries exceeded. Returning the query ID itself, instead of the parent ID...")
        return efo_id #return the same id if there is an error
    
    resp_txt = response.text
    data = json.loads(resp_txt)
    term = data['_embedded']['terms'][0]
    return term.get('obo_id', '')

def fetch_data(efo_id, specific_id=None, page_number=0, retries=10):
    logging.info(f"\nFetching data for {efo_id}, page numer {page_number}, retries left {retries}...\n\n")
    url = f"http://www.ebi.ac.uk/ols4/api/ontologies/efo/children?id={efo_id}&size=500&page={page_number}"
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        logging.error(f"Error occurred: Received status code {response.status_code} instead of 200.")
        if retries > 0:
            return fetch_data(efo_id, specific_id, page_number, retries - 1)
        logging.error("Max retries exceeded. Exiting...")
        return []

    data = json.loads(response.text)
    terms_data = process_data(data, specific_id)

    # If there are more pages, fetch them and combine the results
    if data['page']['totalPages'] > page_number + 1:
        terms_data += fetch_data(efo_id, specific_id, page_number + 1)

    return terms_data

def process_data(data, specific_id=None):
    terms_data = []
    if '_embedded' not in data or 'terms' not in data['_embedded']:
        return terms_data
    for term in data['_embedded']['terms']:
        oboID = term.get('obo_id', '')

        if specific_id is not None and oboID != specific_id:
            continue

        id9 = [r for r in term['annotation'].get('has_dbxref', []) if "ICD9" in r] if 'annotation' in term else []
        id10 = [r for r in term['annotation'].get('has_dbxref', []) if "ICD10" in r] if 'annotation' in term else []
        terms = [term.get('label', '')] + term.get('synonyms', [])
        if bool(term.get('term_replaced_by', '')):
            terms.append(term['term_replaced_by'])

        if oboID.startswith("EFO:"):
            studies = get_studies(efo_id=oboID.replace("EFO:", "EFO_"))
            terms += list(studies.studies['diseaseTrait.trait'].str.lower().unique())
        terms = list({t.lower() for t in terms})

        node = {'oboID': oboID, 'terms': terms, 'ICD9': id9, 'ICD10': id10, 'children': []}

        if specific_id is not None or term.get('has_children', False):
            node['children'] = fetch_data(oboID) #don't pass specific_id here, we want all children this point on

        terms_data.append(node)

    return terms_data

def find_term(ontology_tree, term):
    for node in ontology_tree:
        if term in node['terms']:
            return [node['oboID']]
        path = find_term(node['children'], term)
        if path is not None:
            return [node['oboID']] + path
    return None
def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default="EFO:0004503")
    
    return parser

def main():    
    parser = getARGSParser()
    args, _ = parser.parse_known_args() 

    efo_id = args.id.replace("_",":")
    print(efo_id)

    type_tag = efo_id.split(":")[0]
    idprefix = type_tag if type_tag != "EFO" else ""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', filename=f'/group/glastonbury/soumick/downloads/OLS/{efo_id.replace(f"{type_tag}:", f"ontology_tree_{idprefix}")}.log', filemode='w')
    logging.info('Startring new execution...')

    parent_id = get_parent(efo_id)
    ontology_tree = fetch_data(parent_id, specific_id=efo_id)
    with open(f'/group/glastonbury/soumick/downloads/OLS/{efo_id.replace(f"{type_tag}:", f"ontology_tree_{idprefix}")}.pkl', 'wb') as f:
        pickle.dump(ontology_tree, f)
    #write it as json  file
    with open(f'/group/glastonbury/soumick/downloads/OLS/{efo_id.replace(f"{type_tag}:", f"ontology_tree_{idprefix}")}.json', 'w') as f:
        json.dump(ontology_tree, f)
    
    # # for example, let's say you want to find the term "term_to_find"
    # term_to_find = "term_to_find"  
    # path = find_term(ontology_tree, term_to_find)
    
    # if path is not None:
    #     print(f"The path to '{term_to_find}' is: {path}")
    # else:
    #     print(f"Term '{term_to_find}' was not found in the ontology.")

if __name__ == "__main__":
    main()
