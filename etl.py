# etl functions

import os
from pathlib import Path
import json

import pandas as pd
import numpy as np
from scipy import stats as sps

from datetime import datetime

from collections import OrderedDict, defaultdict

import matplotlib as mpl
from matplotlib import pyplot as plt


def save_file(outfile, s, replace=True):
    """
    Save file or replace it if replace=True &
    write data into a file with extension json as json.
    """
    
    if replace:
        if Path(outfile).exists():
            Path(outfile).unlink()
        
    if isinstance(s, dict) or Path(outfile).suffix=='.json':
        import json
        with open(outfile, 'w') as f:
            f.write(json.dumps(s))   
    else:
        if len(s):
            with open(outfile, 'w') as f:
                f.write(s)
                

# Workflow documentation; Save the raw dataframe info prior to cleanup to file:
def save_df_info(df, outfile, show=False):
    
    with open(outfile, 'w') as f:
        df.info(buf=f)
        
    if show:
        with open(outfile, 'r') as f:
            info_out = f.read()
        print(info_out)


def get_data_year_range(client, dataset_id, date_field, year1, year2, content_type='csv',
                        lim=500_000, local_file=None, replace=False):
    
    if replace:
        if local_file is not None:
            qry = f"date_extract_y({date_field}) between '{year1}' and '{year2}'"
            results = client.get(dataset_id, where=qry, limit=lim,
                                 content_type=content_type)
            # todo: add check on results, 
            
            df = pd.DataFrame(results)
            df.to_csv(local_file)
            return df
        else:
            print('Missing file name for saving.')
  
    if local_file is not None:
        if Path(local_file).exists():
            return pd.read_csv(local_file, index_col=0)


def get_data_year(client, dataset_id, date_field, year, content_type='csv', 
                  lim=500_000, local_file=None, replace=False):
    
    if replace:
        if local_file is not None:
            qry = f"date_extract_y({date_field})='{year}'"
            results = client.get(dataset_id, where=qry, limit=lim, 
                                 content_type=content_type)
            # todo: add check on results
            df = pd.DataFrame(results)
            df.to_csv(local_file)
            return df
        else:
            print('Missing file name for saving.')
            
    if local_file is not None:
        if Path(local_file).exists():
            return pd.read_csv(local_file, index_col=0)            
    
    
def despine(ax, which=['top', 'right']):
    for side in which:
        ax.spines[side].set_visible(False)
        
def format_labels(x, pos):
    return '{0:,}'.format(int(x/1000))   

        
def save_pic(plt, fname=None, transparent=True, replace=False):
    if fname is None:
        return "No file name??"
    if not fname:
        return "No file name??"

    found = Path(fname).exists()
    if found:
        if replace:
            Path(fname).unlink()
            plt.savefig(fname, transparent=transparent)
    else:
        plt.savefig(fname, transparent=transparent)
                    
                    
def show_yearly_counts(df, what="NYPD Arrests",
                           fname=None, savepic=False, 
                           fitline=False, replace=False,
                           fsize=(6, 4), transp=True):
    
    fig, ax1 = plt.subplots(1, 1, figsize=fsize)
    # Assume df has 2 (first) cols: year, count
    x = df[df.columns[0]].values
    y = df[df.columns[1]].values
        
    pct_diff = (y[-1] - y[0])/y[0]
    lbl = f'{what} total\n'
    lbl += f'%Change over period: {pct_diff:.0%}'
    
    ax1.set_title(what + " - Yearly totals")
    ax1.plot(x, y, 'bo', label=lbl)
    ax1.set_xticks(x)
    
    offset = 50_000
    ax1.set_ylim(0, y.max()+offset)
    
    if fitline:
        what = what.lower()
        
        m, b, r_val, p_val, std_err = sps.linregress(x, y)
        # r_val: Coeff. of determination
        
        lbl_fit = f"Line fit ~ {m:,.0f} (yearly {what}) * year "
        lbl_fit += f"+ b={b:,.0f}\n"
        lbl_fit += r"$             R^{2}$= " + f"{r_val:.2f}"
        ax1.plot(x, m * x + b,
                 linewidth=1, linestyle='--',
                 label=lbl_fit)
        
        # for autofmt: rotate the dates for longer ones:
        #rot = 0
    else:
        rot = 30
        fig.autofmt_xdate(rotation=rot)
    
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_labels))
    despine(ax1)
    
    ylocs = ax1.get_yticks()
    ymin = y.min()
    for i in range(len(ylocs)):
        #if i >1:
        if ylocs[i] > ymin - offset:
            ax1.hlines(ylocs[i], x[0], x[-1], 
                       linestyle=':',
                       linewidth=1,
                       alpha=0.5)

    ax1.set_ylabel("Total (thousands)")
    ax1.legend(mode='expand', frameon=False, facecolor="white",
               bbox_to_anchor=(0.1, 0.1, 0.4, 0.35),
               labelspacing=0.75)

    if fname is not None and savepic:
        save_pic(plt, fname=fname, transparent=transp, replace=replace)
    
    plt.show();
    
    
# ... METADATA ...........................................................
global ID_MVD_collisions
ID_MVD_collisions = 'qiz3-axqb'

global ID_311
ID_311 = 'erm2-nwe9'

global ID_Arrests
ID_Arrests = '8h9b-rp9u'

ID_NYS_AADT = ''

def get_rows_from_metadata(metadata, ID):
    """
    MAY NOT WORK FOR ALL DATASETS
    returns value of 'not_null' key in cahedContents for each column.
    """
    rows = -1  #something went wrong if output
    
    for c in metadata['columns']:
        
        if ID == ID_MVD_collisions:
            if c['name'].isupper():   
                # => field names in ID_MVD_collisions; non-upper case are not in the dataset
                if c['name'] == 'UNIQUE KEY':
                    try:
                        rows = int(c['cachedContents']['not_null'])
                    except:
                        rows = 0
                    break
                    
        elif ID == ID_NYS_AADT:
            if c['name'] == 'Region':
                try:
                    rows = int(c['cachedContents']['not_null'])
                except:
                    rows = 0
                break

        elif ID == ID_Arrests:
            if  c['name'].lower() == 'arrest_key':
                try:
                    rows = int(c['cachedContents']['not_null'])
                except:
                    rows = 0
                break
        else:
            if c['name'].lower() == 'unique key':
                try:
                    rows = int(c['cachedContents']['not_null'])
                except:
                    rows = 0
                break
            
    return rows


def get_type_dict_from_meta(meta_NYC):
    col_types = ( [[meta_NYC['columns'][c]['fieldName'], meta_NYC['columns'][c]['dataTypeName']] 
                   for c, v in enumerate(meta_NYC['columns']) ][:-5] )

    data_types_d = dict()
    for i, c in enumerate(col_types):
        if (c[0] == 'date') | (c[0] == 'time'):
            data_types_d[c[0]] = object    # could not pass it a datetime or 'Timestamp'???
        elif (c[0] == 'latitude') | (c[0] == 'longitude'):
            data_types_d[c[0]] = np.float
        elif c[1] == 'number':
            data_types_d[c[0]] = np.int
        else:
            data_types_d[c[0]] = object

    return data_types_d


def get_dataset_span_from_meta(meta_df, date_field):
    """
    meta_df :: output of get_dataset_metainfo(meta_NYC, ID_MVD_collisions);
               where meta_NYC :: metadata from the sodapy/Socrata client
    """
    #date_fields = ['DATE', 'Created Date']
    newest_date = meta_df.loc[date_field, 'Largest'][:10]
    oldest_date = meta_df.loc[date_field, 'Smallest'][:10]

    yr_newest = int(newest_date[:4]) 
    yr_oldest = int(oldest_date[:4])
    yr_span =  yr_newest - yr_oldest
    
    return [yr_oldest, yr_newest]        


def get_dataset_colophon(metadata, client, version, rows):
    """
    Function to retrieve the "dataset colophon"
    """
    display_dict = OrderedDict()
    display_dict['Dataset Name'] = metadata['name']
    display_dict['Dataset Identifier'] = metadata['id']
    display_dict['Total Rows'] = '{:,}'.format(rows)
    display_dict['Source Domain'] = client.domain
    display_dict['Created'] = datetime.utcfromtimestamp(metadata['createdAt']).strftime('%F %T')
    display_dict['Last Updated'] = datetime.utcfromtimestamp(metadata['rowsUpdatedAt']).strftime('%F %T')
    display_dict['Category'] = metadata['category']
    display_dict['Attribution'] = metadata['attribution']
    display_dict['Owner'] = metadata['owner']['displayName']
    display_dict['Endpoint Version'] = version
    
    return display_dict


def show_colophon(d, save_html=False):
    """
    To display the website box "About This Dataset" from a dictionnary, with option to save as html.
    Parameters:
    
        d:  Dictionnary (from metadata)
        save_html (default=False):  save the string as html with name 'ID_colophon.html'.
    
    """
    from IPython.display import display, Markdown #, HTML
    #s = '<div class = "colophon"><table>'
    s = '<table>'
    
    for k, v in d.items():
        s = s + '<tr>'
        td = '<td><strong>' + k + '</strong>:</td><td>' + str(v) + '</td>'
        s = s + td + '</tr>'
     
    # Last row like footer:
    s = (s + '<td><tiny><em>As at (report date):</em></tiny></td><td><tiny><em>'
           + datetime.strftime(datetime.today(), '%Y-%m-%d')
           + '</em></tiny></td>')
            
    s = s + '</table>' #'<div>'
    # TODO  pass file name instead, save in intermediate?
    if save_html:
        outfile = './data/raw/' + d['Dataset Identifier'] + 'colophon.html'
        save_file(outfile, s)
        
    return Markdown(s) #HTML(s) #display(HTML(s))


def get_dataset_metainfo(metadata, ID):
    """
    Function to retrieve "dataset info" from the metadata call: min/max and null values in each column.
    As this function parses the metada of any dataset, it can be used in a quality control analysis.
    Parameters:
        metadata:      sopy.Socrata metadata object
    Output:
        A pandas dataframe with an added columns for the % of Null values.
    """
    
    cols_dict = OrderedDict()

    for c in metadata['columns']:
        name_dict = dict()
        
        if ID == ID_MVD_collisions:
            if c['name'].isupper():   # => field names in ID_MVD_collisions; non-upper case are not in the dataset
                
                for f in ['Largest', 'Smallest', 'Null', 'Not null']:
                    if 'cachedContents' in c.keys():  # this is where the info we need is
                        if f == 'Not null':
                            ff = 'not_null'
                        else:
                            ff = f.lower()

                        name_dict[f] = (c['cachedContents'][ff] if ff in c['cachedContents'].keys() else '')
                    else:
                        name_dict[f] = ''
            else:
                continue
                        
        #elif ID == ID_NYS_AADT:
        else:
            for f in ['Largest', 'Smallest', 'Null', 'Not null']:
                if 'cachedContents' in c.keys():  # this is where the info we need is
                    if f == 'Not null':
                        ff = 'not_null'
                    else:
                        ff = f.lower()

                    name_dict[f] = (c['cachedContents'][ff] if ff in c['cachedContents'].keys() else '')
                else:
                    name_dict[f] = ''
            
        cols_dict[c['name']] = name_dict

    df = pd.DataFrame(cols_dict).T
    # Add a column showing the % of null values:
    df['Null%'] = [ np.round(100 * np.float(x)/np.float(y), 1) if (len(x) and len(y)) else np.nan for x, y in zip(df['Null'], df['Not null']) ]
    # Reorder the columns:
    df = df[['Largest', 'Smallest', 'Null', 'Not null', 'Null%']]

    return df

# ... DATA ................................................................
# X:: using Socrata client
def get_raw_yearly_countX(cli, ID, year_list, date_field, 
                         lim=500_000, local_file=None, replace=False):
    if replace:
        if local_file is not None:
            yrly_counts = []

            for yr in year_list:
                qry = f"count(ARREST_KEY) where date_extract_y({date_field})='{yr}'"
                tot = cli.get(ID, select=qry, limit=lim)
                yrly_counts.append((yr, int(tot[0]['count_ARREST_DATE'])))

            df = pd.DataFrame(yrly_counts, columns=['Year', 'Total'])
            df.to_csv(local_file)
            return df
        else:
            print('Missing file name for saving.')
            
    if local_file is not None:
        if Path(local_file).exists():
            return pd.read_csv(local_file, index_col=0)
        else:
            print(f'File {local_file} not found.')
        
def get_raw_yearly_count_date_splitX(cli, ID, year, date_field, 
                                    lim=500_000, local_file=None, replace=False):
    if replace:
        if local_file is not None:
            qry = f"date_extract_m({date_field}) as Month,"
            qry += f"date_extract_dow({date_field}) as DoW "
            qry_where = f"date_extract_y({date_field})='{year}'"
                
            results = None
            results = cli.get(ID, select=qry, where = qry_where,
                              content_type='json', limit=lim)

            df = pd.DataFrame(results, columns=['Month', 'DoW'])
            df.to_csv(local_file)
        else:
            print('Missing file name for saving.')
# X .........................................................................            
def get_raw_age_groups(cli, ID, local_file=None, replace=False):
    """
    Meant for cli_NYC, ID='8h9b-rp9u'
    """
    if replace:
        if local_file is not None:
            results = cli.get(ID, select="count(ARREST_KEY),AGE_GROUP", group='AGE_GROUP')  
            save_file(local_file, results, replace=replace)
        else:
            print('Missing file name for saving.')
            
    if local_file is not None:
        if Path(local_file).exists():
            with open(local_file) as fh:
                results = json.load(fh)
            return results
        else:
            print(f'File {local_file} not found.')


def get_nyc_results(client, dataset_id, year, rows, raw_filename):
    qry = 'date_extract_y(DATE)={}'.format(year)
    results = client.get(dataset_id, where=qry, limit=rows)
    # todo: add check on results
    
    #Convert json output to pandas DataFrame & save as csv:
    df = pd.DataFrame.from_records(results)
    df.to_csv(raw_filename)
    return df
    
    
def load_csv_df(local_csv, data_types_d=''):
    # todo: get info from meta to obtain the cols datatype
    # -> to use in dtype parameter
    if len(data_types_d) == 0:
        df = pd.read_csv(local_csv, index_col=0)
    else:
        df = pd.read_csv(local_csv, dtype=data_types_d, index_col=0)

    return df 


def load_xtab_df(fname, colindex_name=None, idx_col=0, na_to_str=True):
    df = pd.read_csv(fname, index_col=idx_col)
    if na_to_str:
        df.replace(np.nan, '', inplace=True)
    df.columns.name = colindex_name
    return df


def load_arrests_totals(fname):
    df = pd.read_csv(fname, header=1)
    
    df.dropna(axis=0, inplace=True)
    cols = df.columns.values
    cols[0] = 'Precincts'
    df.columns.name = 'Arrests Totals'
    df.columns = cols
    df.set_index('Precincts', inplace=True)

    return df

def make_xtab(infile_xt_data, outfile_xtab, replace=False):
    """
    Create and save xtab to csv
    """
    if replace:
        df = pd.read_csv(infile_xt_data, index_col=0)
        df = df.astype({'Month':int, 'DoW':int})

        df['Day'] = df['DoW'].map({0:'Sun', 1:'Mon', 
                                   2:'Tue', 3:'Wed', 
                                   4:'Thu', 5:'Fri',
                                   6:'Sat'})

        # group:
        dfg = df.groupby(['Month', 'DoW', 'Day'])['DoW'].count()

        # use dow to sort, then drop:
        dfgu = dfg.unstack(level=0).sort_values(['DoW'])
        dfgu.index = dfgu.index.droplevel('DoW')

        dfgu.to_csv(outfile_xtab)

        
def get_period_yearly_count(data, year_list, local_file=None, replace=False):
    if replace:
        if local_file is not None:
            yrly_counts = [ (yr, data[data.Year==yr].shape[0]) for yr in year_list ]

            df = pd.DataFrame(yrly_counts, columns=['Year', 'Total'])
            df.to_csv(local_file)
            return df
        else:
            print('Missing file name for saving.')
            
    if local_file is not None:
        if Path(local_file).exists():
            return pd.read_csv(local_file, index_col=0)
        else:
            print(f'File {local_file} not found.')
            
            
def get_yearly_count_YMD(data, year, local_file=None, replace=False):
    if replace:
        if local_file is not None:
            data.loc[data.Year==year, ['Month', 'DoW']].to_csv(local_file)
        else:
            print('Missing file name for saving.')
    else:
        if local_file is not None:
            if Path(local_file).exists():
                return pd.read_csv(local_file, index_col=0)
            else:
                print(f'File {local_file} not found.') 
                

# ... REPORTING ..........................................
def add_div_around_html(div_html_text, output_string=False, div_style="{width: 75%}"):
    """
    Wrap an html code str inside a div.
    div_style: whatever follows style= within the <div>
    
    Behaviour with `output_string=True`:
    The cell is overwritten with the output string (but the cell mode is still in 'code' not 'markdown')
    The only thing to do is change the cell mode to Markdown.
    If `output_string=False`, the HTML/md output is displayed in an output cell.
    """
    div = f"""<div style="{div_style}">{div_html_text}</div>"""
    if output_string:
        return div
    else:
        return Markdown(div)
    
    
def ordinal(value):
    """
    Source: http://code.activestate.com/recipes/576888-format-a-number-as-an-ordinal/
    
    Converts zero or a *postive* integer (or their string 
    representations) to an ordinal value.

    >>> for i in range(1,13):
    ...     ordinal(i)
    ...     
    u'1st'
    u'2nd'
    u'3rd'
    u'4th'
    u'5th'
    u'6th'
    u'7th'
    u'8th'
    u'9th'
    u'10th'
    u'11th'
    u'12th'

    >>> for i in (100, '111', '112',1011):
    ...     ordinal(i)
    ...     
    u'100th'
    u'111th'
    u'112th'
    u'1011th'

    """
    try:
        value = int(value)
    except ValueError:
        return value

    if value % 100//10 != 1:
        if value % 10 == 1:
            ordval = u"%d%s" % (value, "st")
        elif value % 10 == 2:
            ordval = u"%d%s" % (value, "nd")
        elif value % 10 == 3:
            ordval = u"%d%s" % (value, "rd")
        else:
            ordval = u"%d%s" % (value, "th")
    else:
        ordval = u"%d%s" % (value, "th")

    return ordval


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: orange' if v else '' for v in is_max]


def highlight_min(s):
    '''
    highlight the minimum in a Series yellow.
    '''
    is_min = s == s.min()
    return ['background-color: #98fb98' if v else '' for v in is_min]


def highlights_styled(df):
    return df.style.apply(highlight_max).apply(highlight_min)


def big_int_style(df):
    return df.style.format("{:,}")
    
def show_bar_plot(df, ax, bar_kind='bar',
                  lgd_bbox=None,
                  lgd_mode=None,
                  fname=None, transp=True, 
                  savepic=False, replace=False,
                   **kwargs):
    """
    Wrapper to customize pandas barplot.
    """
    assert bar_kind in ['bar', 'barh']

    df.plot(kind=bar_kind, ax=ax, **kwargs)

    xlbls = ax.xaxis.get_ticklabels()
    ax.xaxis.set_ticklabels(xlbls, fontweight='bold')
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_labels))
    ax.set_ylabel("Total (thousands)")

    # customizing the legend removes the pandas format (i.e. title needs resetting)
    ncols = 1
    legt = None
    if df.columns.name:
        legt = df.columns.name
        
    bbox = (0.9, 0.6, 0.4, 0.35) if lgd_bbox is None else lgd_bbox
    if lgd_mode == 'expand':
        handles, labels = ax.get_legend_handles_labels()
        ncols = len(handles)
        
    ax.legend(title=legt,
              ncol=ncols, 
              frameon=False, facecolor="white",
              bbox_to_anchor=bbox,
              mode=lgd_mode,
              labelspacing=0.75)

    fig = plt.gcf()
    fig.autofmt_xdate(rotation=0, ha='center')
    despine(ax)
    
    #plt.tight_layout(pad=.9)
    
    if fname is not None and savepic:
        save_pic(plt, fname=fname, transparent=transp, replace=replace)
    
    return ax

        
def color_top1(val):
    """
    Apply the css property `'color: red'` if
    val contains ordinal string '1st'.
    """
    color = 'red' if '1st' in val else 'black'
    return f'color: {color:s}'


def bold_top1(val):
    """
    Apply the css property `'font-weight: bold'` if
    val contains ordinal string '1st'.
    """
    fontweight = 'bold' if '1st' in val else 'normal'
    return f'font-weight: {fontweight:s}'


def combine_all_yearly_top5(dflist, years, fname):
    """
    Transform combined yearly top5 data into compact table
    showing ordinal ranking with count over the period (years).
    """
    # check
    assert len(dflist) == len(years)
    
    # dynamical order: as per last year:
    lastY = years[-1]
    common_top5_desc = (pd.concat(dflist, sort=None).drop('count', axis=1)
                        .sort_values('pd_desc')
                        .pd_desc.unique())

    top5_data = defaultdict(list)
    # Transform data to show ordinal ranking with count
    for desc in common_top5_desc:
        yrs_data = list()
        for i, yr in enumerate(years):
            df = dflist[i]
            # get the index; index+1 = rank
            idx = df[df.pd_desc==desc].index.tolist()
            if len(idx):
                idx = idx[0]
                tot = df.loc[idx, 'count']
                idx += 1
                ox = ordinal(idx)
                yrs_data.append(f'{ox}:   {tot:,}')
            else:
                yrs_data.append(np.nan)
            del df
            
            top5_data[desc] = yrs_data   

    df = pd.DataFrame(top5_data, dtype=int).T
    
    # Current order is alpha: sort as per lastY & prevY order:
    # temp for sorting: cols as int, will need str for formatter
    df.columns = years
    df.reset_index(inplace=True)
    df.replace(np.nan, 'z', inplace=True)
    df.sort_values([lastY, lastY-1], inplace=True)
    df.replace('z','', inplace=True)
    
    df.set_index('index', inplace=True)
    df.index.name = 'Arrests description in any yearly Top 5 ranking'
    df.columns = [str(y) for y in years]
    
    df.to_csv(fname)
    return df


def top5_style(df):
    return df.style.applymap(color_top1).applymap(bold_top1)    


def get_evol_stats(fname):
    # fname = top5_evol_final
    
    df_evol = load_xtab_df(fname)
    
    stats_dict = dict()
    
    # Percentage difference:
    
    # arrrest type with max, min pct_diff:
    stats_dict['min_delta_desc'] = df_evol.loc[:, df_evol.columns[-1]].idxmin()
    stats_dict['min_delta'] = df_evol.loc[:, df_evol.columns[-1]].min()
    stats_dict['max_delta_desc'] = df_evol.loc[:, df_evol.columns[-1]].idxmax()
    stats_dict['max_delta'] = df_evol.loc[:, df_evol.columns[-1]].max()
    
    # n Decreasing?
    msk = df_evol.loc[:, df_evol.columns[-1]] < 0
    stats_dict['count_decreasing_delta'] = msk.value_counts()[True]
    
    # How large a change, n above 5%?
    msk = df_evol.loc[:, df_evol.columns[-1]].abs() > 0.05
    stats_dict['count_delta_above_5pct'] = msk.value_counts()[True]
    
    # Variability: comparison across arrest type

    evol_stats18 = df_evol.T.loc[df_evol.T.index[:-1], :].describe()
    
    # which arrest types are the least/most variable in the last 4 years?
    stats_dict['min_std_desc'] = evol_stats18.loc['std',:].idxmin() 
    min_dev = evol_stats18.loc['std',:].min()
    stats_dict['min_std'] = min_dev

    stats_dict['max_std_desc'] = evol_stats18.loc['std',:].idxmax()
    max_dev = evol_stats18.loc['std',:].max()
    stats_dict['max_std'] = max_dev

    # max dev multiples of min dev
    stats_dict['min_max_std_fold'] = np.round(max_dev/(min_dev + 0.00001), 0)

    return stats_dict