import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(data, num_var:list[str], bin, title, xlab):

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.histplot(
        data[num_var], 
        color='gold', 
        binwidth=bin,
        alpha=1,
        zorder=2,
        linewidth=2,
        edgecolor='black',
        shrink=0.7,
        ax=ax
    )

    plt.title(title, fontsize=24, y=0.9, fontweight='bold', fontfamily='monospace')
    ax.set_xlabel(xlab, fontsize=9, fontfamily='monospace')
    ax.set_ylabel('count', fontsize=9, fontfamily='monospace')

    ax.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='black', dashes=(2,7))
    ax.tick_params(axis='both', labelsize=8)
    for direction in ['top','right','left']:
        ax.spines[direction].set_visible(False)

    plt.tight_layout()
    return plt.show()



def plot_distribution(data, num_var:list[str], title, xlab):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.kdeplot(
        x=data[num_var],
        ax=ax,
        shade=True,
        color='gold',
        alpha=0.6,
        zorder=1,
        linewidth=3,
        edgecolor='black'
    )

    mean_value = data[num_var].mean()
    ax.axvline(mean_value, color='olive', alpha=0.7, linestyle='--', linewidth=2, zorder=2)
    ax.text(mean_value, ax.get_ylim()[1] * 0.3, f'mean: {mean_value:.2f}', fontsize=10, fontfamily='monospace', ha='center')

    plt.title(title, fontsize=20, fontweight='bold', fontfamily='monospace')
    ax.set_xlabel(xlab, fontsize=9, fontfamily='monospace')
    ax.set_ylabel('percent area', fontsize=9, fontfamily='monospace')

    ax.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='black', dashes=(2,7))
    ax.tick_params(axis='both', labelsize=8)
    for direction in ['top','right','left']:
        ax.spines[direction].set_visible(False)

    plt.tight_layout()
    return plt.show()


def plot_counts(data, var_group: str, var_to_count: str, title: str):
    df_count = data.groupby(var_group)[var_to_count].count().reset_index()
    df_count = df_count.rename(columns={var_to_count: 'count'})

    fig, ax = plt.subplots(figsize=(10, 6))
    palette1 = ['wheat' for _ in range(len(df_count))]
    max_n_index = df_count['count'].idxmax()
    palette1[max_n_index] = 'gold'

    # Create the bar plot
    sns.barplot(x=var_group, y='count', data=df_count, palette=palette1, ax=ax)

    plt.title(title, fontsize=15, fontweight='bold', fontfamily='monospace', loc='center')
    ax.set_ylabel('count', fontsize=9, fontfamily='monospace')
    ax.set_xlabel(var_group, fontsize=9, fontfamily='monospace')

    ax.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='black', dashes=(2,7))
    ax.tick_params(axis='both', labelsize=8)
    for direction in ['top','right','left']:
        ax.spines[direction].set_visible(False)

    plt.tight_layout()
    return plt.show()


def plot_boxplot(data, x, y, xlabels:list, title, xlab, ylab):
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.boxplot(
        x=x, 
        y=y, 
        data=data, 
        ax=ax, 
        order=[True, False],
        linewidth=3,
        flierprops={'marker': '.'},
        showcaps=False,
        palette=['gold', 'wheat']
    )
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(xlab, fontsize=9, fontfamily='monospace')
    ax.set_ylabel(ylab, fontsize=9, fontfamily='monospace')
    ax.set_title(title, fontsize=15, fontweight='bold', fontfamily='monospace', loc='center')

    ax.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='black', dashes=(2,7))
    ax.tick_params(axis='both', labelsize=8)
    for direction in ['top','right','left']:
        ax.spines[direction].set_visible(False)

    plt.tight_layout()
    return plt.show()

def plot_scatter(data, xvar, yvar, title, xlab, ylab):
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.scatterplot(
        data=data,
        x=xvar,
        y=yvar,
        ax=ax,
        color='goldenrod',
        alpha=0.7,
        zorder=2
    )
    plt.title(title, fontsize=18, y=0.9, fontweight='bold', fontfamily='monospace')
    ax.set_xlabel(xlab, fontsize=9, fontfamily='monospace')
    ax.set_ylabel(ylab, fontsize=9, fontfamily='monospace')

    ax.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='black', dashes=(2,7))
    ax.tick_params(axis='both', labelsize=8)
    for direction in ['top','right','left']:
        ax.spines[direction].set_visible(False)

    plt.tight_layout()
    return plt.show()
