import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("changes updated")

import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_dangerously_set_inner_html    

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import statsmodels
import statsmodels.api as sm
import plotly.graph_objects as go

from pytrends.request import TrendReq
from pandas import read_csv
from pandas.io.formats.style import Styler
from IPython.display import display
from statsmodels.tools.eval_measures import meanabs
from statsmodels.tools.eval_measures import meanabs
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
from jupyterplot import ProgressPlot

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import base64



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.layers import GaussianNoise

sys.path.insert(0, os.path.dirname(__file__))


class Plotter:

    def plotly_plot(self, industry_count, predicted_df, column, actual_values, predictions, starting_quarter, ending_quarter, annotation, title, summary=None):
        predicted_df["Actual %s Values" % column] /= 1000
        predicted_df["Predicted %s Values" % column] /= 1000
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predicted_df.index, y=predicted_df["Actual %s Values" % column],
                        mode='lines+markers',
                        name='Actual Values - %s' % column,
                        line=dict(width=3)))
        fig.add_trace(go.Scatter(x=predicted_df.index, y=predicted_df["Predicted %s Values" % column],
                        mode='lines+markers',
                        name='Predicted Values - %s' % column,
                        line=dict(width=3)))
        annotation = annotation.replace(',','<br>').replace('for ', 'for<br>')
        fig.add_annotation(x=1, y=predicted_df["Predicted %s Values" % column][-1], xref='paper', xshift=-20,xanchor="left", text=annotation, showarrow=False, font=dict(size=14))
        fig.update_xaxes(showline=True, linewidth=4, linecolor='blue', mirror=True)  
        fig.update_yaxes(showline=True, linewidth=4, linecolor='blue', mirror=True)
        title = title.replace(',','<br>')
        fig.update_layout(
        title= title,
        autosize=False,
        width=700,
        legend=dict(
                orientation="h",
                y = -0.3
            )
        )

        fig.write_image('fig.svg')
        s1=''
        with open('fig.svg', encoding='utf-8') as f:
            s1 = f.read()
        
        s1=s1.replace('height="500"','',1).replace('width="700"','',1).replace('700','800',1).replace('700','800',1)
        
        predicted_df.index.name = "Quarter"
        predicted_df.reset_index(level=0, inplace=True)

        layout = html.Div(children=[
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(s1),
            dbc.Table.from_dataframe(predicted_df),
            html.Div(dash_dangerously_set_inner_html.DangerouslySetInnerHTML(summary.as_html())) if summary else html.Div()
            ],
            className='wrapper huge',
            style={'padding': '10px', 'margin-top': '20px'}
        )

        return layout

    def matplotlib_plot(self, industry_count, predicted_df, column, actual_values, predictions, starting_quarter, ending_quarter, annotation, title, summary=None):
        plt.figure(figsize=(20,7.5))
        plt.rcParams["axes.edgecolor"] = "blue"
        plt.rcParams["axes.linewidth"] = 5
        font = {'weight' : 'bold',
            'size'   : 20}
        plt.rc('font', **font)
        plt.ylabel("In Thousand Dollars")
        predicted_df["Actual %s Values" % column] /= 1000
        predicted_df["Predicted %s Values" % column] /= 1000
        plt.plot(predicted_df.index, predicted_df["Actual %s Values" % column], label="Actual Values - %s" % column, color="red", linewidth=5)
        plt.plot(predicted_df.index, predicted_df["Predicted %s Values" % column], label="Predicted Values - %s" % column, color="blue", linewidth=5)
        plt.xticks(predicted_df.index, rotation=90)
        plt.legend(fontsize=18)
        plt.title(title, fontsize=20, fontweight="bold")
        font = {'weight' : 'bold',
            'size'   : 22}
        plt.rc('font', **font)
        plt.annotate(annotation, (predicted_df.index[-1], predicted_df.iloc[-1, 1]), fontsize=14)  
        print('\n\033[1m\033[4mTable %d: %s\033[0m' % (industry_count, column))
        predicted_df.index.name = "Quarter"
       
        temp_file = column +".png"
        plt.savefig(temp_file, bbox_inches = "tight", dpi=300)
        
        # Embed the result in the html output.
        data = base64.b64encode(open(temp_file, "rb").read()).decode('ascii')

        img = html.Img(
        width='100%',
        src = 'data:image/png;base64,{}'.format(data)
        )
        
        os.remove(temp_file)

        predicted_df.reset_index(level=0, inplace=True)
        
        layout = html.Div(children=[
        img,
        html.H6(children='Table %s: %s' % (industry_count, column)),
        dbc.Table.from_dataframe(predicted_df),
        html.Div(dash_dangerously_set_inner_html.DangerouslySetInnerHTML(summary.as_html())) if summary else html.Div()
        ],
        className='wrapper huge',
        style={'padding': '10px', 'margin-top': '20px'}    
        )

        return layout


class Modeller:

    def __init__(self, ploter, plot_type, show_trend=False):
        self.weighing_method = "expalmon" # or "beta"
        self.rf_lag = 1
        self.pytrend = TrendReq(hl='en-US')
        self.ploter = ploter
        self.layouts = []
        self.show_trend = show_trend
        self.plot = ploter.plotly_plot
        if plot_type == 'MAT':
            self.plot = ploter.matplotlib_plot

    def border_color_green(self, val):
        return ["border : solid 3px green" for x in val]

    # this function requests the latest data from Google Trends
    def request_google(self, catno=73, kw_list = [], geo = "SG"):
        """
        Calls Googletrend data.
        Arguments: 
            category number. for example in the google link:
            https://trends.google.com/trends/explore?cat=68&geo=SG
            the cat is 68
            
        Returns:
            pandas series of google requested data
        """
        self.pytrend.build_payload(kw_list=kw_list,
                        cat = catno,
                        timeframe = 'all', 
                        # or'today 5-y', 'today 3-m', 'all' 
                        geo=geo)
        
        interest_over_time_df = self.pytrend.interest_over_time().iloc[:,0]
        return(interest_over_time_df)

    def next_quarter(self, prev_quarter):
        year = int(prev_quarter[:4])
        quarter_num = int(prev_quarter[-1])
        if quarter_num >= 4:
            quarter_num = 1
            year = year + 1
        else:
            quarter_num += 1
        return str(year) + "-Q" + str(quarter_num)

    def prev_quarter(self, quarter):
        year = int(quarter[:4])
        quarter_num = int(quarter[-1])
        if quarter_num == 1:
            quarter_num = 4
            year = year - 1
        else:
            quarter_num -= 1
        return str(year) + "-Q" + str(quarter_num)

    def get_date_from_quarter(self, quarter):
        year = quarter.split("-")[0]
        q = quarter.split("-")[1][1]
        if q == "1":
            month = "01"
        elif q == "2":
            month = "04"
        elif q == "3":
            month = "07"
        elif q == "4":
            month = "10"
        return (year + "-" + month + "-" + "01")

    def model(self, starting_quarter, ending_quarter):

        quarters = []
        quarter = starting_quarter

        while quarter != self.next_quarter(ending_quarter):
            quarters.append(quarter)
            quarter = self.next_quarter(quarter)

        starting_quarter = quarters[0]
        ending_quarter = quarters[-1]

        with open("result (New NSA).json", "r") as f:
            data = json.loads(f.read())

        main_df = pd.DataFrame()    

        for dic in data["Level2"]:
            main_df.loc[dic["quarter"], dic["level_2"]] = dic["value"]

        main_df.loc[ending_quarter, :] = np.nan
        main_df = main_df.loc["2004-Q1":, :]

        predictions_df = pd.DataFrame(index=quarters)
        
        flag = 0
        double_flag = 0
        industry_count = 1

        # starting loop for every Level 2 Item
        for column in main_df.columns:
            
            trends_params = trends_dict[column]
            
            trends1 = self.request_google(kw_list=[trends_params[0][0]], geo=trends_params[0][1])
            trends2 = self.request_google(kw_list=[trends_params[1][0]], geo=trends_params[1][1])
            
            trends1 = trends1["2004-01-01":self.get_date_from_quarter(quarters[-1])]
            trends2 = trends2["2004-01-01":self.get_date_from_quarter(quarters[-1])]

            # if column in series_to_predict:
            #   print("Google Trends Data: \n")
            #   display(pd.concat([trends1, trends2], axis=1).style.apply(border_color_green).set_table_styles([{'selector':'th','props':[('border','3px solid green'), ('color','red')]}]))

            # trends1 = np.log(trends1).dropna()
            # trends2 = np.log(trends2).dropna()
            
            actual_values = []
            predictions = []
            
            end_date = starting_quarter
            end_date_one_minus = self.prev_quarter(end_date)
            
            # print(column)
            if column in series_to_predict:
                print("\nPredicting for %s now.\n" % column)

            index = 0
            while True:
                
                if end_date in quarters:
                    train = main_df.loc['2004-Q1':end_date, column]
                    train_logged = train.copy()
                    minimum = min(train_logged)
            
                    if column in series_to_predict:
                        if end_date == starting_quarter:
                            # print("\nTraining Dataframe")
                            # display_train = train_logged.copy(deep=True)
                            # display_train.index.name = "Quarter"
                            # display(pd.DataFrame(display_train).style.apply(border_color_green).set_table_styles([{'selector':'th','props':[('border','3px solid green'), ('color','red')]}]))
                            print("\n\033[1m\033[4mTitle %d: OLS Out-of-Sample Forecast of %s : %s (Level 2), for Periods %s to %s\033[0m" % (industry_count, column, series_to_predict[column], starting_quarter, ending_quarter))
                            #pp = ProgressPlot(line_names=["Actual Values", "Predicted Values"], line_colors=["#FF3333", "#3336FF"], width=1250,x_label="Quarter", x_iterator=False, x_lim= [1, len(quarters)])
                    #######################################################################################################################################################################
                    # F Test (Used the OLS model to do F Test, because current model produces p value = nan, where both string method and array method are used.)
                    # Results are carried out with the help of examples given on http://www.statsmodels.org/v0.10.0/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.f_test.html
                    #######################################################################################################################################################################
                    # print("\nF Test:\n")
                    # formula = "rsi ~ "
                    # hypotheses = ""
                    # for name in trends_columns_names:
                    #   formula += name + "+ "
                    #   hypotheses += "(" + name + " = 0), " #This part here is a loop which mentions all the google trends
                    # formula += "cny_b + cny_d + cny_a"
                    # results = ols(formula, test).fit()
                    # #hypotheses += "(cny_b = 0), (cny_d = 0), (cny_a = 0)"
                    # A = np.identity(len(results.params))
                    # print("Result Parameters:", results.params[1:-3])
                    # A = A[1:,:]
                    # print("Hypotheses string method:")
                    # print("Hyptheses:", hypotheses[:-2])
                    # print(results.f_test(hypotheses[:-2]))
                    # print("\nArray method:")
                    # print("Array:", A)
                    # print(results.f_test(A))

                    train_logged = train_logged.dropna()
                    train_logged[end_date] = np.nan
                            
                    train_logged.index = pd.DatetimeIndex(train_logged.index)
                    last_date = train_logged.index[-2]

                    train_trends1 = trends1[:end_date].copy()
                    train_trends2 = trends2[:end_date].copy()

                    """
                    C. TRAIN THE MODEL

                    """
                    """
                    1. Fit the model on train
                    """
                #     train = train.dropna()
                #     train_logged = train_logged.dropna()
                        
                    combined_df = pd.concat([train_logged, train_trends1, train_trends2],axis=1,join='outer').dropna()
                    
                        
                    model = OLS(endog = combined_df.iloc[:-1, :-2], exog=combined_df.iloc[:-1, -2:])  
                    
                    fit_res = model.fit()  
                    
                        
                    """
                    D. DISPLAY PREDICTION RESULTS

                    """


                    # print("\nForecast Results for %s, %s: \n" % (end_date, column))
                    
                    prediction = fit_res.get_prediction(exog=combined_df.iloc[-1:,-2:]).summary_frame()  
                    last_row = prediction.tail(1)
                    value = last_row.iloc[0][0]
                    prediction = value  
                    print('prediction is ', prediction)
                    try:
                        print(predicted_df.iloc[-1, 1])
                    except Exception:
                        pass

                    predictions.append(prediction)
                    actual_values.append(train[end_date])
                    print('Appended predictions is/are: ', predictions)
                    
                    
                        # print('\033[94m\033[1m' + str(prediction) + '\033[0m')

                    if column in series_to_predict:
                        #pp.update(index+1, [[train[starting_quarter:ending_quarter].values[index] / 1000, prediction / 1000]])
                        time.sleep(0.5)
                        index += 1
                
                summary = fit_res.summary() if self.show_trend else None
                
                if end_date == ending_quarter:
                    if column in series_to_predict:
                        print('pp. finalize is ')  
                    #pp.finalize()
                    break
                end_date_one_minus = end_date
                end_date = self.next_quarter(end_date)

                


            # Plotting actual RSI vs Predicted RSI

            predictions_df[column] = predictions
            
            if column in series_to_predict:
                predicted_df = pd.DataFrame({"Actual %s Values" % column: actual_values, "Predicted %s Values" % column: predictions}, index=quarters)
                title = 'Title %d: OLS Out-of-Sample Forecast of %s : %s (Level 2), for Periods %s to %s' % (industry_count, column, series_to_predict[column], starting_quarter, ending_quarter)
                annotation = "Latest Prediction for %s : %s, %s: %f" % (column, series_to_predict[column], predicted_df.index[-1], predicted_df.iloc[-1, 1]/1000)
                self.layouts.append(self.plot(industry_count, predicted_df, column, actual_values, predictions, starting_quarter, ending_quarter, annotation, title, summary=summary))
                industry_count += 1

            
        print("\nGetting Predictions for Level 1 Items:\n")
            
        level_dictionary = {'NA000330Q': {'NA000331Q': 1, 'NA000332Q': 1},
                            'NA000333Q': {'NA000330Q': 1, 'NA000351Q': 1}, 
                            'NA000342Q': {'NA000343Q': 1, 'NA000344Q': 1},
                            'NA000352Q': {'NA000353Q': 1, 'NA000354Q': 1},
                            'NA000374Q': {'NA000342Q': 1, 'NA000352Q': 1},
                            'NA000336Q': {'NA000339Q': 1, 'NA000340Q': 1, 'NA000341Q': 1},
                            'NA000337Q': {'NA000336Q': 1, 'NA000338Q': 1},
                            'NA000335Q': {'NA000337Q': 1, 'NA000373Q': 1},
                            'NA000347Q': {'NA000346Q': 1, 'NA000348Q': 1},
                            'NA000349Q': {'NA000347Q': 1, 'NA000350Q': 1},
                            'NA000334Q': {'NA000333Q': 1, 'NA000335Q': 1, 'NA000349Q': 1, 'NA000374Q': 1},
                            }


        with open("result (New NSA).json", "r") as f:
            data = json.loads(f.read())

        level1_df = pd.DataFrame()  

        for dic in data["Level1"]:
            level1_df.loc[dic["quarter"], dic["level_1"]] = dic["value"]
            
        level1_df.loc[ending_quarter, :] = np.nan

        for column, array in level_dictionary.items():
            predicted_df = pd.DataFrame({"Actual %s Values" % column: level1_df.loc[quarters, column]}, index=quarters)
            predicted_df["Predicted %s Values" % column] = 0
            
            
            for item in array:
                if item == "NA000342Q":
                    predicted_df["Predicted %s Values" % column] -= predictions_df[item] * array[item]
                else:
                    predicted_df["Predicted %s Values" % column] += predictions_df[item] * array[item]
                    
            # fig1, ax1 = plt.subplots()
            # ax1.pie(pie_values_array, labels=array, autopct='%1.1f%%',
            #     shadow=True, startangle=90)
            # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            # plt.show()
                    
            if column not in predictions_df.columns:
                predictions_df[column] = predicted_df["Predicted %s Values" % column]
                    
            if column in series_to_predict:
                print("\n\033[1m\033[4mTitle %d: OLS Out-of-Sample Forecast of %s : %s (Level 1), for Periods %s to %s" % (industry_count, column, series_to_predict[column], starting_quarter, ending_quarter))
                annotation = "Latest Prediction for %s, %s: %f" % (column, predicted_df.index[-1], predicted_df.iloc[-1, 1]/1000)
                title = 'Title %d: OLS Out-of-Sample Forecast of %s : %s (Level 1), for Periods %s to %s' % (industry_count, column, series_to_predict[column], starting_quarter, ending_quarter)
                self.layouts.append(self.plot(industry_count, predicted_df, column, actual_values, predictions, starting_quarter, ending_quarter, annotation, title, summary=summary))
                industry_count += 1

        return self.layouts


starting_quarter = [
                    '2015-Q1',
                    '2015-Q2',
                    '2015-Q3',
                    '2015-Q4',
                    '2016-Q1',
                    '2016-Q2',
                    '2016-Q3',
                    '2016-Q4',
                    '2017-Q1',
                    '2017-Q2',
                    '2017-Q3',
                    '2017-Q4',
                    '2018-Q1',
                    '2018-Q2',
                    '2018-Q3',
                    '2018-Q4',
                    '2019-Q1',
                    '2019-Q2',
                    '2019-Q3',
                    '2019-Q4',
                    '2020-Q1',
                    '2020-Q2',
                    '2020-Q3',
                    ]

ending_quarter = [
                  '2015-Q1',
                  '2015-Q2',
                  '2015-Q3',
                  '2015-Q4',
                  '2016-Q1',
                  '2016-Q2',
                  '2016-Q3',
                  '2016-Q4',
                  '2017-Q1',
                  '2017-Q2',
                  '2017-Q3',
                  '2017-Q4',
                  '2018-Q1',
                  '2018-Q2',
                  '2018-Q3',
                  '2018-Q4',
                  '2019-Q1',
                  '2019-Q2',
                  '2019-Q3',
                  '2019-Q4',
                  '2020-Q1',
                  '2020-Q2',
                  '2020-Q3'
                  ]

series_names = {'NA000330Q': 'Mandarin',
                'NA000331Q': 'Apple',
                'NA000332Q': 'Watermelon',
                'NA000333Q': 'Jackfruit',
                'NA000334Q': 'Papaya',
                'NA000335Q': 'Grapefruit',
                'NA000336Q': 'Lemon',
                'NA000337Q': 'Apricot',
                'NA000338Q': 'Orange',
                'NA000339Q': 'Pear',
                'NA000340Q': 'Cherry',
                'NA000341Q': 'Strawberry',
                'NA000342Q': 'Kiwi',
                'NA000343Q': 'Nectarine',
                'NA000344Q': 'Grape',
                'NA000346Q': 'Mango',
                'NA000347Q': 'Melon',
                'NA000348Q': 'Blueberry',
                'NA000349Q': 'Coconut',
                'NA000350Q': 'Pomegranate',
                'NA000351Q': 'Carambola',
                'NA000352Q': 'Pineapple',
                'NA000353Q': 'Plum',
                'NA000354Q': 'Banana',
                'NA000373Q': 'Raspberry',
                'NA000374Q': 'Lime'}

series_to_predict = {}



trends_dict = {'NA000331Q': [['shopping', 'US'], 
                             ['employment', 'US']],
                'NA000332Q': [['shopping', 'US'], 
                              ['employment', 'US']],
                 'NA000338Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000339Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000340Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000341Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000343Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000344Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000346Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000348Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000350Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000351Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000353Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000354Q': [['shopping', 'US'], 
                               ['employment', 'US']],
                 'NA000373Q': [['shopping', 'US'], 
                               ['employment', 'US']]
              }


# def get_dates(starting_date, ending_date):  
#   dates = []
#   date = starting_date
#   while str(date) != str(ending_date):
#     dates.append(date)
#     year = int(date.split("-")[0])
#     month = int(date.split("-")[1])
#     month += 1
#     if month == 13:
#       month = 1
#       year += 1
#     if month > 9:
#       month = str(month)
#     else:
#       month = "0" + str(month)
#     date = str(year) + "-" + month + "-01" 
#     print(date)
#   dates.append(ending_date)
#   return dates


dropdown_industry = []
for i in series_names:
    m = {'label':i,'value':i}
    dropdown_industry.append(m)

drop_dates_start = []
for i in starting_quarter:
  m = {'label':i,'value':i}
  drop_dates_start.append(m)

drop_dates_end = []
for i in ending_quarter:
  m = {'label':i,'value':i}
  drop_dates_end.append(m)


external_css = [
    'https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0-rc.2/css/materialize.min.css',
    'https://fonts.googleapis.com/icon?family=Material+Icons',
    'https://codepen.io/muhnot/pen/bKzaZr.css', 
    'https://cdnjs.cloudflare.com/ajax/libs/animate.css/3.7.2/animate.min.css',
    'https://res.cloudinary.com/dlmqmx0oc/raw/upload/v1609646491/style_vi1ar3.css'
]

external_js = [
     'https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0-rc.2/js/materialize.min.js',
     'https://code.jquery.com/jquery-3.3.1.min.js',
     'https://codepen.io/muhnot/pen/bKzaZr.js'
]

app = dash.Dash(
    external_scripts=external_js,
    external_stylesheets=external_css,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
  )
server = app.server

review_layout = []

stats_layout = [
    html.Div([
      html.Div(className='col s12 m5', children=dbc.Button("Back", type='submit', id="back", className="col-content green darken-4 white-text", n_clicks=0,style={'marginTop': 10})),
      html.H1('Literature Review and Methodology'),
      html.P('Lorem ipsum dolor sit amet consectetur adipisicing elit. Tempora repellendus temporibus distinctio molestias. Veniam laudantium omnis voluptas dolorem dolor possimus impedit minima, ducimus corrupti quas facere delectus autem aspernatur dignissimos deserunt id saepe odit qui fugit sunt. Veniam, sunt eum?')
      ], className='review', id='review', style={'display':'none'}
    ),
    html.Div([
      html.H2("Trend Predictor: ", style={'color':'white'}),
      html.Div(className='row',children = [
        html.Div(className='col s5 m4 wrapper',children=[
          html.Label('Enter a Start Date', htmlFor='start-date', style={'fontWeight':'bold','fontSize': '1rem', 'color': 'white'}),
          dcc.Dropdown(
          id='start-date',
          options=drop_dates_start,
          style={'marginBottom': 10, 'marginTop': 10, 'color': 'black'}
          )
          ],
          style={'margin': '4px'}),
        html.Div(className='col s5 m4 wrapper',children=[
            html.Label('Enter an End Date', htmlFor='end-date', style={'fontWeight':'bold','fontSize': '1rem','color': 'white'}),
            dcc.Dropdown(
            id='end-date',
            options=drop_dates_end,
            style={'marginBottom': 10, 'marginTop': 10, 'color': 'black'} 
          )
          ],
          style={'margin': '4px'}),
          html.Div(className='col s11 m3 wrapper',children=[
            html.Label('Select Visualization', htmlFor='plot-type', style={'fontWeight':'bold','fontSize': '1rem','color': 'white'}),
            dcc.Dropdown(
              id='plot-type',
              options=[
                  {'label': 'Plotly', 'value': 'PLOT'},
                  {'label': 'Matplotlib', 'value': 'MAT'}
              ],
              value='PLOT',      
              style={'marginBottom': 10, 'marginTop': 10, 'color': 'black'} 
            ) 
          ],
          style={'margin': '4px'}),
        html.Div(className='col s11 wrapper',children=[
          html.Label('Select Series to Predict', htmlFor='dropdown', style={'fontWeight':'bold','fontSize': '1rem', 'color': 'white'}),
          dcc.Dropdown(
          id='dropdown',
          options=dropdown_industry,
          multi=True,
          style={'marginBottom': 10, 'marginTop': 10, 'color': 'black'}
          )
          ],
          style={'margin': '4px'}),
        html.Div(className='col s12', children=[
          html.Center(children=[
            html.Div(className='col s12 m5', children=dbc.Button("Literature Review and Methodology", type='submit', id="show-literature", className="col-content green darken-4 white-text", n_clicks=0,style={'marginTop': 10})),
            html.Div(className='col s12 m3', children=dbc.Button("PREDICT TREND", id="submit-val", className="col-content blue", n_clicks=0,style={'marginTop': 10})),
            html.Div(className='col s12 m4', children=dbc.Button("SHOW TREND PARAMETERS", id="show-trend", className="col-content orange", n_clicks=0,style={'marginTop': 10})),
          ])
        ]),
        html.Div(className='col s12',children=[
          dcc.Loading(
                      id="loading-2",
                      children=[html.Div(id='output')],
                      type="default",
                      fullscreen=True
                  )
          ])
      ])
    ], className='stats', id='stats', style={'display':'block'})
]



theme_colors=[
  {'label':'Purple', 'value': 'purple darken-4'},
  {'label':'Blue',   'value': 'blue darken-4'},
  {'label':'Green',  'value': 'green darken-4'},
  {'label':'Black',  'value': 'black'},
  {'label':'Brown',  'value': 'brown darken-4'},
]

app.layout = html.Div([
    html.Div(
      html.Div([
        html.Label('Select Theme', htmlFor='color-picker', style={'color':'black'}),
        dcc.Dropdown( 
          options=theme_colors, id='color-picker', value='purple darken-4', clearable=False)],
      className='col s7 m3 wrapper', style={'padding':'4px'}
      ),
      className='container row wrapper-orange', style={'padding':'4px'}
    ),
    html.Div(id='page-content', className='container purple darken-4',children=[
      *stats_layout
    ], style={'borderRadius': '5px', 'padding':'4px', 'marginTop':'10px', 'color': 'white'})
  ]
)

@app.callback(Output(component_id='page-content', component_property='className'),
              [
                Input(component_id='color-picker', component_property='value'),
              ])
def update_background_color(value):
  class_name = 'container {} {}'
  wrapper_theme = 'wrapper-orange'
  if 'purple' in value:
    wrapper_theme = 'wrapper-green'
  name = class_name.format(value, wrapper_theme)
  return name

@app.callback([
    Output(component_id='review', component_property='style'),
    Output(component_id='stats', component_property='style')
  ],
  [
    Input(component_id='show-literature', component_property='n_clicks'),
    Input(component_id='back', component_property='n_clicks')
  ]
  )
def hide_review(l_clicks, b_clicks):
  changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
  print(changed_id)
  if 'show-literature' in changed_id:
    return [{'display': 'block'}, {'display': 'none'}]
  else:
    return [{'display': 'none'}, {'display': 'block'}]

@app.callback(
    dash.dependencies.Output('output', 'children'),
    [
      dash.dependencies.Input(component_id='submit-val', component_property='n_clicks'),
      dash.dependencies.Input(component_id='show-trend', component_property='n_clicks'),
      dash.dependencies.State(component_id='start-date', component_property='value'),
      dash.dependencies.State(component_id='end-date', component_property='value'),
      dash.dependencies.State(component_id='dropdown', component_property='value'),
      dash.dependencies.State(component_id='plot-type', component_property='value')
    ])
def update_output(clicks, show_trend, starting_quarter, ending_quarter, industry_names, plot_type):
    trend = False
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-val' in changed_id:
        show_trend = False
    elif 'show-trend' in changed_id:
        trend = True
        clicks += 1
    if industry_names != None and starting_quarter != None and clicks > 0:
        print('starting model')
        for s in industry_names:
            series_to_predict[s] = series_names[s]
        print(series_to_predict)
        ploter = Plotter()
        model = Modeller(ploter,plot_type, show_trend=trend)
        layouts = model.model(starting_quarter, ending_quarter) 
        return layouts

if __name__ == '__main__':
    app.run_server(debug=True,port='8080', host="0.0.0.0")
