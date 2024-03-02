
def dictionary_maker(lm,k=15):

    out = {}
    out['pathway_names'] = []
    out['pathway_rationales'] = []

    for i in range(k):
        out['pathway_names'].append(lm[f'pathway_{i+1}'])
        out['pathway_rationales'].append(lm[f'rationale_{i+1}'])

    return out


def self_consistency(dict_list,normalize=False):

    out = {}

    for d in dict_list:
        
        for pathway,rationale in zip(d['pathway_names'],d['pathway_rationales']):
            if pathway not in out.keys():
                out[pathway] = {}
                out[pathway]['count'] = 1
                out[pathway]['rationales'] = [rationale]
            else:
                out[pathway]['count'] += 1
                out[pathway]['rationales'].append(rationale)

    #devide rationales by count
    if normalize:
        for key in out.keys():
            out[key]['count'] = out[key]['count']/len(dict_list)

    return out