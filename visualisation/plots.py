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
