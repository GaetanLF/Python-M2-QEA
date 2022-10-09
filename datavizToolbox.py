#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:24:10 2022

@author: SCHMIDT, Julia & LE FLOCH, Gaëtan
    
Please run the notebook index.ipynb to use this homemade module.    

"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
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
        
            INPUT: the new object storing the class, the original dataframe,
        the target variable.
        
            OUTPUT: The new class, fully defined.
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
                
    
            
    def showScatters(self,y=None):
        """
            This function generates all scatter plots with respect to the 
        target variable or another variable denoted by y.
        
            INPUT: One should only call this method in Python after running
        __init__. If necessary, please specify y the variable of interest.
        
            OUTPUT: All scatter plots wrapped in a widget.
        """
        if y==None: # Hence, the y axis denotes the target
            cols = self.predictors
            y = self.target
        else:
            cols = self.df.columns.tolist()
            cols.remove(y)

        @interact
        def read_values(
        predictor=widgets.Dropdown(
            description="Select :", value=cols[0], options=cols
        )):
            fig = px.scatter(self.df, x = predictor, y = y)
            go.FigureWidget(fig.to_dict()).show()
            
        
            
    def showBoxplots(self,listCategorical):
        """
            This function generates all box plots with respect to the 
        target variable.
        
            INPUT: The list of categorical variables to consider
        
            OUTPUT: All boxplots wrapped in a widget.
        """
        @interact
        def read_values(
        predictor=widgets.Dropdown(
            description="Select :", value=listCategorical[0], options=listCategorical
        )):
            fig = px.box(self.df, x = predictor, y = self.target)
            go.FigureWidget(fig.to_dict()).show()
            
    def showDensities(self,by=None,disp_hist=False):
        """
            This function shows all density plots, with a potential division
        by a given categorical variable.
        
            INPUT: the categorical to be used for the division, and the option
        to display the histogram or not.
        
            OUTPUT: All density plots wrapped in a widget.
        """
        cols = self.df.select_dtypes(np.number).columns.tolist() # We should only take quantitative ones.
        if by == None:
            @interact
            def read_values(
            variable=widgets.Dropdown(
                description="Select :", value=cols[0], options=cols
            )):
                fig = ff.create_distplot([self.df[variable]],group_labels=[variable],show_hist=disp_hist)
                go.FigureWidget(fig.to_dict()).show()
        elif by.lower() not in [x.lower() for x in self.df.columns.tolist()]:
            raise KeyError("The variable by is not in the dataframe")
        else:
            groups = self.df[by].unique().tolist()
            @interact
            def read_values(
            variable=widgets.Dropdown(
                description="Select :", value=cols[0], options=cols
            )):
                data=[]
                for modality in groups:
                    data.append(self.df[variable].loc[self.df[by] == modality].tolist())
                fig = ff.create_distplot(data,group_labels=groups,show_hist=disp_hist)
                go.FigureWidget(fig.to_dict()).show()
                
        

        
                
    
        

    
