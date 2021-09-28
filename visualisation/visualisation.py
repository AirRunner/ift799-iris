def hist_plot(df, var, cats='species'):
    from itertools import combinations
    from operator import itemgetter
    import matplotlib.pyplot as plt
    from seaborn import histplot, color_palette
    
    colours = ["blue", "orange", "green"]
    palette = dict(zip(colours, color_palette()[0:3]))
    col = lambda c1, c2: itemgetter(c1, c2)(palette)
    
    species = list(df[cats].unique())
    species.reverse()
    
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"Class distributions for {var}")
    
    for i, (cla, cols) in enumerate(zip(species, combinations(colours, 2))):
        histplot(df[df[cats] != cla], x=var, multiple="stack", hue=cats, bins=15, ax=axs[i], palette=col(*cols))
    plt.plot()


def run_pca(df, cats=None, scale=False, n_comps=None, projection=None, plot=False, title=''):
    from sklearn.decomposition import PCA
    from pandas import DataFrame

    cats = df.columns[-1] if cats is None else cats
    x = df.drop(cats, axis=1).to_numpy()
    y = df[cats]

    if scale:
        from sklearn.preprocessing import StandardScaler
        x = StandardScaler().fit_transform(x)
    x_fit = x

    if projection is not None:
        df_scaled = DataFrame(x)
        df_scaled[cats] = y
        x_fit = df_scaled[y == projection].drop(cats, axis=1)

    n_comps = x.shape[1] if n_comps is None else int(n_comps)

    pca = PCA(n_components=n_comps)
    pca.fit(x_fit)
    transformed = pca.transform(x)

    if plot:
        from seaborn import pairplot
        from matplotlib.pyplot import show
        df_trans = DataFrame(transformed, columns=[f"PC{i+1}" for i in range(n_comps)])
        df_trans[cats] = y
        pplt = pairplot(df_trans, hue=cats)
        pplt.fig.suptitle(title, y=1.01)
        show()

    return transformed, pca.components_, pca.explained_variance_
