import json
import argparse
from glob import glob
        
def find_term(ontology_tree, term):
    paths = []
    for node in ontology_tree:
        if term in node['terms']:
            paths.append([node['oboID']])
        child_paths = find_term(node['children'], term)
        for path in child_paths:
            paths.append([node['oboID']] + path)
    return paths

def get_ontology_forrest(root="/group/glastonbury/soumick/downloads/OLS"):
    trees = glob(f'{root}/ontology_tree_*.json')
    ontology_forrest = []
    for tree in trees:
        with open(tree, 'r') as f:
            ontology_tree = json.load(f)
            ontology_forrest += ontology_tree
    return ontology_forrest
        
def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default="EFO:0004294")
    
    return parser

def main():    
    parser = getARGSParser()
    args, _ = parser.parse_known_args() 

    ontology_forrest = get_ontology_forrest()

    # for example, let's say you want to find the term "term_to_find"
    term_to_find = "IVF"
    path = find_term(ontology_forrest, term_to_find)

    root = [p[0] for p in path]
    root = list(set(root))
    print(f"Roots: {root}")

    if path is not None:
        print(f"The path to '{term_to_find}' is: {path}")
    else:
        print(f"Term '{term_to_find}' was not found in the ontology.")

if __name__ == "__main__":
    main()
