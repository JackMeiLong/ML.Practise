# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 09:30:19 2016

@author: meil
"""
import pandas as pd
import numpy as np

df=pd.DataFrame({'key':np.arange(5),'value':[np.nan,'a','b','c',np.nan]})

ser=pd.Series({'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000})

ser=ser.map(lambda x:x+0.5)