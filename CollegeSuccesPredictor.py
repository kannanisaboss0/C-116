#---------------------------------CollegeSuccesPredictor.py---------------------------------#
'''
Importing Modules:
-LogisticRegression (LogReg) :-sklearn.linear_model
-StandardScaler :-sklearn.preprocessing
-accuray_score (a_s) :-sklearn.metrics
-train_test_split (tts) :-sklearn.model_selection
-pandas (pd)
-plotly.graph_objects (go)
-plotly.express (px)
-statistics (st)
-numpy (np)
-time (tm)
'''

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score as a_s
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import statistics as st
import numpy as np
import time as tm


#Defining a function to create a scatter from tthe stipulated data
def CreateScatterGraphFromData(list_x_arg,list_y_arg,label_arg_1,label_arg_2,title_supplement_arg,list_color_arg):
  color_list_param=[]

  for value in list_color_arg:
    #Assessing the value of each variable in the list and assigning a color according to the value
    #Case-1
    if(value==0):
      color_list_param.append("red")

    #Case-2
    else:
      color_list_param.append("green")  

  scatter_param=go.Figure(go.Scatter(x=list_x_arg,y=list_y_arg,mode="markers",marker=dict(color=color_list_param)))

  scatter_param.update_layout(title="{} and {} :-{}".format(label_arg_1,label_arg_2,title_supplement_arg))

  scatter_param.update_yaxes(title_text=label_arg_2)
  scatter_param.update_xaxes(title_text=label_arg_1)

  scatter_param.show()


#Defining a function to differentiate the the data for the purposes of training and testing
def SegregateDataAsTrainAndTest(factors_arg,results_arg):
  factor_train_param,factor_test_param,result_train_param,result_test_param=tts(factors_arg,results_arg,test_size=0.25,random_state=0,train_size=0.75)
  
  return factor_train_param,factor_test_param,result_train_param,result_test_param


#Defining a function to train the the data to correlate the data on the basis of linear regression
def TrainValuesForLineaarRegressionAndCreateScatterGraph(df_arg_1,df_arg_2,df_arg_3,label_arg):
  lin_x=np.array(df_arg_1)
  lin_y=np.array(df_arg_2)
  result=np.array(df_arg_3)

  average_list=[]

  for value in range(len(lin_x)):
    average=(lin_x[value]+lin_y[value])/2
    average_list.append(average)

  slope,intercept=np.polyfit(average_list,result,1)

  y=[]

  for dep_var in average_list:
    y_value=dep_var*slope+intercept
    y.append(y_value)

  scatter_graph=px.scatter(x=average_list,y=result,color=result,title="{} and {} :-{} (Linear Regression)".format(stat_choice_1,stat_choice_2,label_arg))
 
  scatter_graph.update_layout(shapes=[dict(type="line",x0=min(average_list),x1=max(average_list),y0=min(y),y1=max(y))])
  
  scatter_graph.update_xaxes(title_text=stat_choice_1+" and "+stat_choice_2+" (Average)")
  scatter_graph.update_yaxes(title_text="Chance of Admit")
  
  scatter_graph.show()

  dict_object={"x":average_list,"y":y}

  correlation=np.corrcoef(dict_object["x"],dict_object["y"])
  correlation=correlation[0,1]

  correlation_percentage=correlation*100
  correlation_squared_percentage=(correlation**2)*100

  print("The veracity of the data {}%".format(round(correlation_percentage,2)))
  print("{}% of differences in '{}' can be explained by the average of '{}' and '{}'".format(round(correlation_percentage,2),"Chance of Admit",stat_choice_1,stat_choice_2))
  
  return slope,intercept


#Defining a function to predict the y values using x values provided by the user through logistic regression
def PredictDataForLogisticRegression(stat_choice_1_arg,stat_choice_2_arg):
  x_param_1=float(input("Please enter the value of '{}':".format(stat_choice_1_arg)))
  x_param_2=float(input("Please enter the value of '{}':".format(stat_choice_2_arg)))

  user_test_param=SS.fit_transform([[x_param_1,x_param_2]])

  y_value_param=lr.predict(user_test_param)

  print("The chance of the student having completed research with '{}' and '{}' equal to {} and {} is {}".format(stat_choice_1_arg,stat_choice_2_arg,x_param_1,x_param_2,y_value_param[0]))
  
  further_prediction=input("Continue further prediction?(:-Yes or No)")

  #Verifying whether the user wants to conduct further prediction of logistic regression
  #Case-1
  if(further_prediction=="yes" or further_prediction=="Yes"):
    PredictDataForLogisticRegression(stat_choice_1_arg,stat_choice_2_arg)

  #Case-2
  else:
    print("Request Accepted")  


#Defining a function to predict the y values using x values provided by the user through linear regression
def PredictDataForLinearRegression(slope_arg,intercept_arg,stat_choice_1_arg,stat_choice_2_arg):
  x_param_1=float(input("Please enter the value of '{}':".format(stat_choice_1_arg)))
  x_param_2=float(input("Please enter the value of '{}':".format(stat_choice_2_arg)))

  x_list_param=[x_param_1,x_param_2]
  x_list_mean_param=st.mean(x_list_param)

  y_value_param=x_list_mean_param*slope_arg+intercept_arg
  y_value_param_percentage=y_value_param*100

  #Verifying whether the percentage of the y-value is greater than 100% or lesser than 0%
  #Case-1
  if(y_value_param_percentage>100):
    y_value_param_percentage=100

  #Case-2  
  elif(y_value_param_percentage<0):
    y_value_param_percentage=0  

  print("The '{}' when the average is at {} is equal to {} or {}%".format("Chance of Admit",round(x_list_mean_param,2),round(y_value_param,2),round(y_value_param_percentage,2)))
  
  further_prediction=input("Continue further prediction?(:-Yes or No)")

  #Verifying whether the user wants to conduct further prediction of linear regression
  #Case-1
  if(further_prediction=="yes" or further_prediction=="Yes"):
    PredictDataForLinearRegression(slope_arg,intercept_arg,stat_choice_1_arg,stat_choice_2_arg)

  #Case-2
  else:
    print("Request Accepted")  




#Reading data from the file
df=pd.read_csv("data.csv")

#Assigning variables to the imported modules
lr=LogReg()
SS=StandardScaler()

#Introductory statement and user inputs
print("Welcome to CollegeSuccesPredictor.py. We provide linear and multilinear logistic regression for a particular college.")

view_information=input("Do not know what Linear Regression or MultiLinear Logistic Regression is? is?(:- I Know, I Don't Know)")

#Verifying wether the user desires to view information aout multilinear logistic regression or not
#Case-1
if(view_information=="I Don't Know" or view_information=="i don't know" or view_information=="I don't know"):
  print("What is linear regression?")
  tm.sleep(3.2)

  print("Linear regression is a method of prediction where two values are related in the form of a linear equation, y=mx+c.")
  tm.sleep(3.0)

  print("The dependent variable y can be predicted using the equation with the help of the indpendent variable x.")
  tm.sleep(2.3)

  print("Where is linear regression used?")
  tm.sleep(1.8)

  print("Linear regression is the most common type of data prediction.")
  tm.sleep(2.1)

  print("It is used in several fields, such as:")
  tm.sleep(1.2)

  print("1. Predicting the value of a product by corporations to modify their sales plans accordingly.")
  tm.sleep(2.3)

  print("2. Estimating demographic data by demographers to prepare and inform the government accordingly.")
  tm.sleep(2.3)

  print("3. Estimating values in sceintific equations and calculus by the scientific community.")
  tm.sleep(2.3)

  print("Linear regression harbors much more uses and possibilities.")
  tm.sleep(2.4)
  
  print("To know more about linear regression, visit:'https://en.wikipedia.org/wiki/Linear_regression' for more")
  tm.sleep(2.8)

  print("What is MultiLinear Logistic Regression?")
  tm.sleep(2.3)

  print("MultiLinear Logistic Regression is method of prediction where a linear or logistic function depends on several other factors(other than the independent and dependent variables")
  tm.sleep(3.4)

  print("External factors are also integrated into the regression, improving the relation of the regression with reality.")
  tm.sleep(2.3)

  print("However, the number of external factors is inversely proportional to the accuracy of the regression.")
  tm.sleep(2.5)

  print("This signifies that regression accuracy diminishes when the number of external factors increase and vice-versa.")
  tm.sleep(3.4)

  print("MultiLinear Logistic Regression can be used in place of Logistic Regression to increase the reality of the regression, with cost of its accuracy.")
  tm.sleep(3.4)

  print("To know more about linear regression, visit:'https://en.wikipedia.org/wiki/Multinomial_logistic_regression' for more")
  tm.sleep(2.8)


print("Loading Data...")
tm.sleep(2.3)

stat_list_1=["Unusable_Element","GRE Score","TOEFL Score","University Rating","SOP","LOR ","CGPA"]
stat_count_1=0

for stat_1 in stat_list_1[1:]:
  stat_count_1+=1
  print("{}:{}".format(stat_count_1,stat_1))

stat_input_1=int(input("Please enter the index of the perfromance sataisitc desired to be the x-axis:"))
stat_choice_1=stat_list_1[stat_input_1]


stat_list_2=["Unusable_Element","GRE Score","TOEFL Score","University Rating","SOP","LOR ","CGPA"]
stat_count_2=0

for stat_2 in stat_list_2[1:]:
  stat_count_2+=1
  print("{}:{}".format(stat_count_2,stat_2))

stat_input_2=int(input("Please enter the index of the perfromance sataisitc desired to be the y-axis:"))
stat_choice_2=stat_list_2[stat_input_2]


#Verifying whether the user inputs are not equal or equal
#Case-1
if(stat_input_1!=stat_input_2):
  df_research=df["Research"].tolist()

  CreateScatterGraphFromData(df[stat_choice_1],df[stat_choice_2],stat_choice_1,stat_choice_2,"Original Data(Logistic Regression)",df["Research"].tolist())

  df_factors_log=df[[stat_choice_1,stat_choice_2]]
  df_result_log=df[["Research"]]


  factor_train_log,factor_test_log,result_train_log,result_test_log=SegregateDataAsTrainAndTest(df_factors_log,df_result_log)

  factor_train_standardized_log=SS.fit_transform(factor_train_log)
  factor_test_standardized_log=SS.fit_transform(factor_test_log)


  lr=LogReg(random_state=0)
  lr.fit(factor_train_standardized_log,result_train_log)

  predicted_value_log=lr.predict(factor_test_standardized_log)
  veracity_log=a_s(result_test_log,predicted_value_log)

  veracity_percentage_log=veracity_log*100
  print("The data is {}% accurate".format(veracity_percentage_log))


  CreateScatterGraphFromData(factor_train_log[stat_choice_1],factor_train_log[stat_choice_2],stat_choice_1,stat_choice_2,"Trained Data (Logistic Regression)",result_train_log["Research"].tolist())
  CreateScatterGraphFromData(factor_test_log[stat_choice_1],factor_test_log[stat_choice_2],stat_choice_1,stat_choice_2,"Tested Data (Logistic Regression)",result_test_log["Research"].tolist())

  prediction_log=input("Predict data for logistic regression?(:-Yes or No)")

  #Verifying the user's choice to continue prediction for logistic regression
  #Case-1
  if(prediction_log=="Yes" or prediction_log=="yes"):
    PredictDataForLogisticRegression(stat_choice_1,stat_choice_2)
  
  #Case-2
  else:
    print("Request Accepted")


  df_stat_1=df[stat_choice_1].tolist()
  df_stat_2=df[stat_choice_2].tolist()
  df_result=df["Chance of Admit "].tolist()

  df_stat_1_train=factor_train_log[stat_choice_1].tolist()
  df_stat_2_train=factor_train_log[stat_choice_2].tolist()
  df_result_train=df["Chance of Admit "].tolist()

  df_stat_1_test=factor_test_log[stat_choice_1].tolist()
  df_stat_2_test=factor_test_log[stat_choice_2].tolist()
  df_result_test=df["Chance of Admit "].tolist()


  slope_lin,intercept_lin=TrainValuesForLineaarRegressionAndCreateScatterGraph(df_stat_1,df_stat_2,df_result,"Original Data")
  slope_lin_train,intercept_lin_train=TrainValuesForLineaarRegressionAndCreateScatterGraph(df_stat_1,df_stat_2,df_result,"Trained Data")
  slope_lin_test,intercept_lin_test=TrainValuesForLineaarRegressionAndCreateScatterGraph(df_stat_1,df_stat_2,df_result,"Tested Data")

  prediction_lin=input("Predict data for linear regression?(:-Yes or No)")

  #Verifying the user's choice to continue prediction for linear regression
  #Case-1
  if(prediction_lin=="Yes" or prediction_lin=="yes"):
    PredictDataForLinearRegression(slope_lin,intercept_lin,stat_choice_1,stat_choice_2)

  #Case-2
  else:
    print("Request Accepted.")  

  #Printing Ending Message
  print("Thank you for using CollegeSuccesPredictor.py.")

#Case-2
else:
  print("Request Terminated.")  
  print("Invalid Values.")
  print("Please refrain from entering the same value.")

  #Printing Ending Message
  print("Thank You for using CollegeSuccessPredictor.py")
#---------------------------------CollegeSuccesPredictor.py---------------------------------#
