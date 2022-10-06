#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:24:10 2022

@author: 
    
Please run the notebook index.ipynb to use this homemade module.    

"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets
from ipywidgets import interact

class dataset:
    """
        The dataset class aims to take a dataset, define a target and
    draw all descriptive statistics into a Python widget.
    
        INPUT : df, a Pandas dataframe
                target, a string denoting a column
    """
    
    def __init__(self,df,target):
        """
            This function is launched at the very beginning of the definition.
            It initializes all variables within the class and checks for common
        errors.
        
            INPUT: the new object storing the class, the original dataframe and
        the target variable.
        
            OUTPUT: The new class, fully defined.
            
            RULE OF THUMB: If the number of unique values is less than the size
        of the dataframe divided by 200, hence it is deemed as a categorical
        variable. /!\ This OF COURSE works on large datasets only.
        """
        
        # Checking errors
        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError("df is not a Pandas dataframe.")
        if not isinstance(target,str):
            raise TypeError("target is not a string variable.")
        if target.lower() not in [col.lower() for col in df.columns.tolist()]:
            raise ValueError("The target is not available in the dataframe.")
            
        self.df = df
        self.target = target.lower()
        self.predictors = [name.lower() for name in self.df.columns.tolist()]
        self.predictors.remove(self.target)
        
        self.dicoVariables = dict()
        for predictor in self.predictors:
            if self.df[predictor].nunique() < self.df.shape[0]/200:
                self.dicoVariables.update({predictor:'categorical'})
            else:
                self.dicoVariables.update({predictor:'quantitative'})
                
    
            
    def showScatters(self):
        """
            This function generates all scatter plots with respect to the 
        target variable.
        
            INPUT: One should only call this method in Python after running
        __init__.
        
            OUTPUT: All scatter plots wrapped in a widget.
        """
        @interact
        def read_values(
        predictor=widgets.Dropdown(
            description="Select :", value=self.predictors[0], options=self.predictors
        )):
            fig = px.scatter(self.df, x = predictor, y = self.target)
            go.FigureWidget(fig.to_dict()).show()
            
    def showBoxplots(self):
        """
            This function generates all box plots with respect to the 
        target variable.
        
            INPUT: One should only call this method in Python after running
        __init__.
        
            OUTPUT: All boxplots wrapped in a widget.
        """
        qualitativePredictors = [key for key,value in self.dicoVariables.items() if value=='categorical']
        @interact
        def read_values(
        predictor=widgets.Dropdown(
            description="Select :", value=qualitativePredictors[0], options=qualitativePredictors
        )):
            fig = px.box(self.df, x = predictor, y = self.target)
            go.FigureWidget(fig.to_dict()).show()

        
                
    
        

    
