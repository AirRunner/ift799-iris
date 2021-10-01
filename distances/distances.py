def distance(v1, v2, metric='minkowski', L=2, cov=None):
    import numpy as np
    
    if metric == "minkowski":
        return np.power(np.power(v1 - v2, L).sum(), 1/L)
    elif metric == "mahalanobis" and cov is not None:
        return np.sqrt(np.dot(np.dot((v1 - v2).transpose(), np.linalg.pinv(cov)), (v1 - v2)))
    else:
        raise NotImplementedError


def intra_class(df, class_name, cats=None, metric='minkowski', cov=None):
    cats = df.columns[-1] if cats is None else cats
    df = df[df[cats] == class_name].drop(cats, axis=1)
    center = df.mean()
    return df.apply(lambda elt: distance(elt, center, metric=metric, cov=cov), axis=1).max()


# Directional inter-class distance
def inter_class(df, source_class, target_class, cats=None, metric='minkowski', cov=None):
    cats = df.columns[-1] if cats is None else cats
    center = df[df[cats] == target_class].drop(cats, axis=1).mean()
    df = df[df[cats] == source_class].drop(cats, axis=1)
    return df.apply(lambda elt: distance(elt, center, metric=metric, cov=cov), axis=1).min()


def pair_distances(df, cats=None):
    import numpy as np
    from pandas import DataFrame
    from itertools import combinations
    
    cats = df.columns[-1] if cats is None else cats
    distances = []
    
    for (class1, class2) in combinations(df[cats].unique(), 2):
        for _ in range(2):
            cov = np.cov(df[df['species'] == class1].drop('species', axis=1).transpose())

            intra_euclid = intra_class(df, class1)
            inter_euclid = inter_class(df, class2, class1)
            
            intra_mahala = intra_class(df, class1, metric="mahalanobis", cov=cov)
            inter_mahala = inter_class(df, class2, class1, metric="mahalanobis", cov=cov)
            
            distances.append([
                class1, class2, intra_euclid, inter_euclid, intra_euclid < inter_euclid,
                intra_mahala, inter_mahala, intra_mahala < inter_mahala
            ])
            class1, class2 = class2, class1
            
    return DataFrame(distances, columns=[
        "class 1", "class 2", "intra_euclid", "inter_euclid", "separated_euclid",
        "intra_mahala", "inter_mahala", "separated_mahala"
    ])
