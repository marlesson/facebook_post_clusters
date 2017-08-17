import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# Probability Distribution Graph
# 
def plot_dist_prob(outpath, samples):
    N_bins = 50

    # make a fit to the samples
    shape, loc, scale = stats.lognorm.fit(samples, floc=0)
    x_fit       = np.linspace(samples.min(), samples.max(), 100)
    samples_fit = stats.lognorm.pdf(x_fit, shape, loc=loc, scale=scale)

    # plot a histrogram with linear x-axis
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.2})
    counts, bin_edges, ignored = ax1.hist(samples, N_bins, histtype='stepfilled', alpha=0.4,
                                          label='histogram')

    # calculate area of histogram (area under PDF should be 1)
    area_hist = ((bin_edges[1:] - bin_edges[:-1]) * counts).sum()

    # plot fit into histogram
    ax1.plot(x_fit, samples_fit*area_hist, label='PDF log-normal', linewidth=2)
    ax1.legend()
    plt.ylabel('Total posts')
    plt.xlabel('Engagement')   

    # equally sized bins in log10-scale and centers
    bins_log10 = np.logspace(np.log10(samples.min()), np.log10(samples.max()), N_bins)
    bins_log10_cntr = (bins_log10[1:] + bins_log10[:-1]) / 2

    # histogram plot
    counts, bin_edges, ignored = ax2.hist(samples, bins_log10, histtype='stepfilled', alpha=0.4,
                                          label='histogram')

    # calculate length of each bin and its centers(required for scaling PDF to histogram)
    bins_log_len = np.r_[bin_edges[1:] - bin_edges[: -1], 0]
    bins_log_cntr = bin_edges[1:] - bin_edges[:-1]

    # get pdf-values for same intervals as histogram
    samples_fit_log = stats.lognorm.pdf(bins_log10, shape, loc=loc, scale=scale)
    
    # pdf-values for centered scale
    samples_fit_log_cntr = stats.lognorm.pdf(bins_log10_cntr, shape, loc=loc, scale=scale)

    # plot fitted and scaled PDFs into histogram
    ax2.plot(bins_log10_cntr, 
         samples_fit_log_cntr * bins_log_cntr * counts.sum(), '-', 
         label='PDF normal', linewidth=2)


    ax2.set_xscale('log')
    ax2.set_xlim(bin_edges.min(), bin_edges.max())
    ax2.legend(loc=3)
    plt.savefig(outpath+'probability_dist.png')
    plt.clf()
       

def plot_silhouette(outpath, clusters, AvgSil):
    # Var IntraGrupo
    fig, ax = plt.subplots(figsize=(15,6), facecolor='w')
    ax.yaxis.grid() # horizontal lines

    plt.plot(clusters, AvgSil)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average silhouette ')
    plt.savefig(outpath+'silhouette_cluster.png')
    plt.clf()

def plot_total_posts_per_cluster(outpath, df):
    # Total de Posts por Cluster
    count = df[['cluster', 'eng']].groupby(['cluster'], as_index=False).count()

    # plot
    fig, ax = plt.subplots(figsize=(15,6), facecolor='w')
    ax.yaxis.grid() # horizontal lines

    plt.xlabel('Cluster')
    plt.ylabel('Total Posts')
    plt.bar(count.cluster, count.eng)
    plt.xticks(range(len(count.cluster)), count.cluster)
    plt.savefig(outpath+'plot_total_posts_per_cluster.png')
    plt.clf()

def plot_avg_eng_per_cluster(outpath, df):
    # Média do engajamento por cluster
    average = df[['cluster', 'eng']].groupby(['cluster'], as_index=False).mean()

    
    # plot
    fig, ax = plt.subplots(figsize=(15,6), facecolor='w')
    ax.yaxis.grid() # horizontal lines

    plt.xlabel('Cluster')
    plt.ylabel('Average engagement')
    plt.bar(average.cluster, average.eng)
    plt.xticks(range(len(average.cluster)), average.cluster)  
    plt.savefig(outpath+'plot_avg_eng_per_cluster.png')
    plt.clf()

def plot_box_avgt_eng_per_cluster(outpath, df):
    # Média do engajamento por cluster
    average = df[['cluster', 'eng']].groupby(['cluster'], as_index=False).mean()

    # Plot Box
    data_plot_log = []
    data_plot = []
    
    for k in range(0, average.cluster.size):
        data_plot_log.append(np.log10(df[df['cluster'] == k].eng))
        data_plot.append(df[df['cluster'] == k].eng)

    # Create a figure instance
    fig = plt.figure(1, figsize=(15, 6))
    
    plt.xlabel('Cluster')
    plt.ylabel('Engagement')

    # Create an axes instance
    ax = fig.add_subplot(111)
    #ax.yaxis.grid() # horizontal lines

    # Create the boxplot
    bp = ax.boxplot(data_plot, showmeans=True, meanline=True, showfliers=False)
    ax.set_xticklabels(average.cluster)
    plt.savefig(outpath+'plot_box_avgt_eng_per_cluster.png')
    plt.clf()

def plot_dist_prob_per_cluster(outpath, df):
    g = sns.FacetGrid(df, col='cluster', col_wrap=4)
    g.map(plt.hist, 'eng', bins=20)
    g.set_axis_labels("Posts", "Engagement")
    g.savefig(outpath+'plot_dist_prob_per_cluster.png')

def plot_word_cloud(outpath, cluster, text):
    wordcloud = WordCloud(background_color="white", width = 1000, height = 500).generate(text)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(outpath+'wordsclound/%s_plot_word_cloud.png' %cluster)
    plt.clf()
    