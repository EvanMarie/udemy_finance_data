import pandas as pd
import numpy as np
import matplotlib as mpl
from IPython.core.display import HTML
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.2f}'.format

bgcolor = '#333333';
text_color = 'white'
innerbackcolor = "#222222";
outerbackcolor = "#333333";
fontcolor = "white"

favorite_cmaps = ['cool', 'autumn', 'autumn_r', 'Set2_r', 'cool_r',
                  'gist_rainbow', 'prism', 'rainbow', 'spring']


# FUNCTIONS: d, p, sp, table_of_contents, display_me, sample_df, see,
#	     list_to_table, div_print, overview, missing_values, fancy_plot

# .......................IMPORTS....................................... #
def pd_np_mpl_import():
    global pd
    global np
    global plt
    global reload

    pd = __import__('pandas', globals(), locals())
    np = __import__('numpy', globals(), locals())
    matplotlib = __import__('matplotlib', globals(), locals())
    plt = matplotlib.pyplot
    importlib = __import__('importlib', globals(), locals())
    reload = importlib.reload


def url_import():
    global urlretrieve
    urllib = __import__('urllib', globals(), locals())
    urlretrieve = urllib.request.urlretrieve

def yf_import():
    global yf
    yf = __import__('yfinance', globals(), locals())

def import_all():
    pd_np_mpl_import()
    url_import()
    yf_import()


# ......................TINY_GUYS....................................... #

def sp(): print('');

def p(x): print(x); sp()

def d(x): display(x); sp()

def pretty(input, label=None, fontsize=3, bgcolor='#444444',
           textcolor="white", width=None
           ):
    from IPython.display import HTML
    input = str(input)

    def get_color(color):
        label_font = [x - 3 for x in [int(num) for num in color[1:]]]
        label_font = "#" + "".join(str(x) for x in label_font)
        return label_font

    if label != None:
        label_font = get_color(bgcolor)
        pretty(label, fontsize=fontsize, bgcolor='#ececec',
               textcolor=label_font, width=width, label=None)

    display(HTML("<span style = 'line-height: 1.5; \
                                background: {}; width: {}; \
                                border: 1px solid text_color;\
                                border-radius: 0px; text-align: center;\
                                padding: 5px;'>\
                                <b><font size={}><text style=color:{}>{}\
                                </text></font></b></style>".format(bgcolor, width,
                                                                   fontsize,
                                                                   textcolor, input)))

def div_print(text, width='auto', bgcolor=bgcolor, text_color=text_color,
              fontsize=2
              ):
    from IPython.display import HTML as html_print

    if width == 'auto':
        font_calc = {6: 2.75, 5: 2.5, 4: 2.5, 3: 3, 2: 4}
        width = str(len(text) * fontsize * font_calc[fontsize]) + "px"

    else:
        if type(width) != str:
            width = str(width)
        if width[-1] == "x":
            width = width
        elif width[-1] != '%':
            width = width + "px"

    return display(html_print("<span style = 'display: block; width: {}; \
						line-height: 2; background: {};\
						margin-left: auto; margin-right: auto;\
						border: 1px solid text_color;\
						border-radius: 3px; text-align: center;\
						padding: 3px 8px 3px 8px;'>\
						<b><font size={}><text style=color:{}>{}\
						</text></font></b></style>".format(width, bgcolor,
                                                           fontsize,
                                                           text_color, text)))

# .......................Time Stamp Converter....................................... #
# Write a function to convert any column in a df that is a timestamp
# to date, hour, and min only
# Find columns that dtype is timestamp
def time_stamp_converter(df):
    def find_timestamp(df):
        timestamp_cols = []
        for col in df.columns:
            if df[col].dtype.name.startswith('datetime64'):
                timestamp_cols.append(col)
        return timestamp_cols
    timestamp_cols = find_timestamp(df)
    for col in timestamp_cols:
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    return df

# .......................DISPLAY_ME........................................ #
def head_tail_vert(df, num, title, bgcolor=bgcolor,
                    text_color=text_color, fontsize=4,
                    intraday=False):
    from IPython.core.display import HTML

    if type(df) != pd.core.frame.DataFrame:
        df = df.copy().to_frame()

    if not intraday:
        df = time_stamp_converter(df.copy())
        if df.index.dtype.name.startswith('datetime64'):
            df.index = df.index.strftime('%Y-%m-%d')
            # df.index = df.index.date

    head_data = "<center>" + df.head(num).to_html()
    tail_data = "<center>" + df.tail(num).to_html()

    print("")
    div_print(f'{title}: head({num})', fontsize=fontsize,
              bgcolor=bgcolor, text_color=text_color)
    display(HTML(head_data))
    print("")
    div_print(f'{title}: tail({num})', fontsize=fontsize,
              bgcolor=bgcolor, text_color=text_color)
    display(HTML(tail_data))
    print("")

def head_tail_horz(df, num, title, bgcolor=bgcolor,
                   text_color=text_color, precision=2,
                   intraday=False, title_fontsize=4,
                   table_fontsize="12px"):

    if type(df) != pd.core.frame.DataFrame:
        df = df.copy().to_frame()

    if not intraday:
        df = time_stamp_converter(df.copy())
        if df.index.dtype.name.startswith('datetime64'):
            df.index = df.index.strftime('%Y-%m-%d')
            # df.index = df.index.date

    div_print(f'{title}', fontsize=title_fontsize,
              bgcolor=bgcolor, text_color=text_color)
    multi([(df.head(num),f"head({num})"),
           (df.tail(num),f"tail({num})")],
          fontsize=table_fontsize, precision=precision,
          intraday=intraday)

# .......................SEE....................................... #

def see(data, title=None, width="auto", fontsize=4,
        bgcolor=bgcolor, text_color=text_color,
        intraday=False):

    pd.options.display.float_format = '{:,.2f}'.format

    if title != None:
        div_print(f"{title}", fontsize=fontsize, width=width,
                  bgcolor=bgcolor, text_color=text_color)

    if isinstance(data, pd.core.frame.DataFrame):
        if not intraday:
            data = time_stamp_converter(data.copy())
            if data.index.dtype.name.startswith('datetime64'):
                data.index = data.index.strftime('%Y-%m-%d')
                # data.index = data.index.date

        display(HTML("<center>" + data.to_html()));
        sp()
    elif isinstance(data, pd.core.series.Series):
        if data.index.dtype.name.startswith('datetime64'):
            data.index = data.index.strftime('%Y-%m-%d')
            # data.index = data.index.date
        display(HTML("<center>" + data.to_frame().to_html()));
        sp()
    else:
        try:
            display(HTML("<center>" + data.to_frame().to_html()));
            sp()
        except:
            pretty(data, title);
            sp()


# .......................FORCE_DF....................................... #
def date_only(data, intraday=False):
    if intraday == False:
        if data.index.dtype == 'datetime64[ns]':
            data.index = data.index.strftime('%Y-%m-%d')
            # data.index = data.index.date
            return data
        else:
            return data
    else:
        return data


def force_df(data, intraday=False):
    if isinstance(data, pd.core.series.Series):
        return date_only(data, intraday=intraday).to_frame()
    elif isinstance(data, pd.core.frame.DataFrame):
        return date_only(data, intraday=intraday)
    else:
        try:
            return pd.Series(data).to_frame()
        except:
            return div_print("The data cannot be displayed.")

# .......................MULTI....................................... #

def multi(data_list, fontsize='15px', precision=2, intraday=False):
    from IPython.display import display_html

    caption_style = [{
        'selector': 'caption',
        'props': [
            ('background', bgcolor),
            ('border-radius', '3px'),
            ('padding', '5px'),
            ('color', text_color),
            ('font-size', fontsize),
            ('font-weight', 'bold')]}]

    thousands = ",";
    spaces = "&nbsp;&nbsp;&nbsp;"
    table_styling = caption_style

    stylers = []
    for idx, pair in enumerate(data_list):
        if len(pair) == 2:
            table_attribute_string = "style='display:inline-block'"
        elif pair[2] == 'center':
            table_attribute_string = "style='display:inline-grid'"
        styler = force_df(data_list[idx][0], intraday=intraday).style \
            .set_caption(data_list[idx][1]) \
            .set_table_attributes(table_attribute_string) \
            .set_table_styles(table_styling).format(precision=precision,
                                                    thousands=thousands)
        stylers.append(styler)

    if len(stylers) == 1:
        display_html('<center>' + stylers[0]._repr_html_(), raw=True); sp();
    elif len(stylers) == 2:
        display_html('<center>' + stylers[0]._repr_html_() + spaces + stylers[1]._repr_html_() + spaces, raw=True); sp();
    elif len(stylers) == 3:
        display_html('<center>' + stylers[0]._repr_html_() + spaces + stylers[1]._repr_html_() + spaces + stylers[
            2]._repr_html_() + spaces, raw=True); sp();
    elif len(stylers) == 4:
        display_html('<center>' + stylers[0]._repr_html_() + spaces + stylers[1]._repr_html_() + spaces + stylers[
            2]._repr_html_() + spaces + stylers[3]._repr_html_() + spaces, raw=True); sp();

# .......................LIST_TO_TABLE....................................... #

def list_to_table(display_list, num_cols, title, width="auto",
                  bgcolor=bgcolor, text_color=text_color
                  ):
    div_print(f"{title}", fontsize=4, width=width,
              bgcolor=bgcolor, text_color=text_color)

    count = 0
    current = '<center><table><tr>'
    length = len(display_list)
    num_rows = round(length / num_cols) + 1

    for h in range(num_rows):
        for i in range(num_cols):
            try:
                current += ('<td>' + display_list[count] + '</td>')
            except IndexError:
                current += '<td>' + ' ' + '</td>'
            count += 1
        current += '</tr><tr>'
    current += '</tr></table></center>'
    display(HTML(current))

# .......................MISSING_VALUES....................................... #

def missing_values(df, bgcolor=bgcolor, text_color=text_color):
    from IPython.display import HTML
    pd.options.display.float_format = '{:,.0f}'.format
    missing_log = []
    for column in df.columns:
        missing_values = df[column].isna().sum()
        missing_log.append([column, missing_values])
    missing = pd.DataFrame(missing_log, columns=['column name', 'missing'])
    div_print(f'Columns and Missing Values', fontsize=3, width="38%",
              bgcolor=bgcolor, text_color=text_color)
    missing = "<center>" + missing.to_html()
    display(HTML(missing))


# ............................FANCY_PLOT....................................... #

def fancy_plot(data, kind="line", title=None, legend_loc='upper right',
               xlabel=None, ylabel=None, logy=False, outerbackcolor=outerbackcolor,
               innerbackcolor=innerbackcolor, fontcolor=fontcolor, cmap='cool',
               label_rot=None
               ):
    import random
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['xtick.color'] = outerbackcolor
    mpl.rcParams['ytick.color'] = outerbackcolor
    mpl.rcParams['font.family'] = 'monospace'
    fig = plt.subplots(facecolor=outerbackcolor, figsize=(13, 7))
    ax = plt.axes();
    if kind == 'line':
        data.plot(kind='line', ax=ax, rot=label_rot, cmap=cmap, logy=logy)
    else:
        data.plot(kind=kind, ax=ax, rot=label_rot, cmap=cmap, logy=logy);
    plt.style.use("ggplot");
    ax.set_facecolor(innerbackcolor)
    ax.grid(color=fontcolor, linestyle=':', linewidth=0.75, alpha=0.75)
    plt.tick_params(labelrotation=40);
    plt.title(title, fontsize=23, pad=20, color=fontcolor);
    plt.ylabel(ylabel, fontsize=18, color=fontcolor);
    plt.xlabel(xlabel, fontsize=18, color=fontcolor);
    plt.xticks(fontsize=10, color=fontcolor)
    plt.yticks(fontsize=10, color=fontcolor)
    if legend_loc is None:
        ax.get_legend().remove()
    else:
        plt.legend(labels=data.columns, fontsize=15, loc=legend_loc,
                   facecolor=outerbackcolor, labelcolor=fontcolor)

# ****************************MINI-PLOT********************************** #
def mini_plot(df, title, ylabel=None, xlabel=None, cmap='cool', kind='line',
              label_rot=None, logy=False, legend_loc=2
              ):
    mpl.rcParams['xtick.color'] = text_color
    mpl.rcParams['ytick.color'] = text_color
    mpl.rcParams['font.family'] = 'monospace'
    fig, ax1 = plt.subplots(figsize=(13, 7), facecolor=outerbackcolor)

    if kind == 'line':
        df.plot(kind='line', ax=ax1, rot=label_rot, cmap=cmap, logy=logy)
    else:
        df.plot(ax=ax1, kind=kind, rot=label_rot, cmap=cmap, logy=logy)
    plt.title(title, color=text_color, size=20, pad=20)
    ax1.grid(color='LightGray', linestyle=':', linewidth=0.5, which='major', axis='both')

    plt.xticks()
    ax1.set_ylabel(ylabel, color=text_color)
    ax1.set_xlabel(xlabel, color=text_color)
    plt.style.use("ggplot");
    ax1.set_facecolor(innerbackcolor)
    if legend_loc is None:
        ax.get_legend().remove()
    else:
        plt.legend(labels=df.columns, fontsize=15, loc=legend_loc,
                   facecolor=outerbackcolor, labelcolor=text_color)
    plt.show()


# ****************************PLOT-BY-DF********************************** #

def plot_by_df(df, sort_param, title, fontsize = 4, num_records=12,
               ylabel=None, xlabel=None, cmap='cool', kind='line',
               label_rot=None, logy=False, legend_loc=2, precision=2,
               thousands=",", intraday=False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from ipywidgets import GridspecLayout

    if not intraday:
        df = time_stamp_converter(df.copy())
        if df.index.dtype.name.startswith('datetime64'):
            df.index = df.index.strftime('%Y-%m-%d')
            # df.index = df.index.date

    out_box1 = widgets.Output(layout={"border": "1px solid black"})
    out_box2 = widgets.Output(layout={"border": "1px solid black"})

    heading_properties = [('font-size', '11px')]
    cell_properties = [('font-size', '11px')]

    dfstyle = [dict(selector="th", props=heading_properties), \
               dict(selector="td", props=cell_properties)]

    with out_box1:
        display(date_only(df.sample(num_records).sort_values(sort_param)).style \
                .format(precision=precision, thousands=thousands) \
                .set_table_styles(dfstyle))

    with out_box2:
        mini_plot(df, title, ylabel=ylabel, xlabel=xlabel, cmap=cmap, kind=kind,
                  label_rot=label_rot, logy=logy, legend_loc=legend_loc)

    grid = GridspecLayout(20, 8)
    grid[:, 0] = out_box1
    grid[:, 1:20] = out_box2
    div_print(f"{title} ({num_records} samples & overall plot)", fontsize=fontsize)
    display(grid)


# *************************MULTI-PLOT-By-DF************************************* #
# MUST PASS list of lists to this function
# [[df, title], [df, title], [df, title], [df, title]]

def multi_plot_by_df(data_list, title, xlabel=None, ylabel=None,
                     legend_loc=2, cmap='cool', num_records=12,
                     sort_param=None, fontsize=3, precision=2, thousands=","):

    div_print(title, fontsize=5)

    for pair in data_list:
        plot_by_df(pair[0], title=pair[1], xlabel=xlabel, fontsize=fontsize,
                   ylabel=ylabel, legend_loc=legend_loc, cmap=cmap,
                   num_records=num_records, sort_param=sort_param,
                   precision=precision, thousands=thousands);sp();

    sp(); sp();
