import pubchempy as pcp

def get_pubchem_id(name):
    results = pcp.get_compounds(name, 'name')

    if len(results) == 0:
        results = pcp.get_substances(name, 'name')

        if len(results) == 0:
            return None,None
        
        else:
            return results[0].sid, 'Substance'

    else:
        return results[0].cid, 'Compound'