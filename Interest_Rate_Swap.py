# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:21:15 2022

@author: Arath Reyes
"""


import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay



# En caso de que la curva de forwards no esté construida hay que hacer Bootstrapping

# Pci = [28,27,28,29,27] 
[29,28,30,31]

def InterpolacionLineal(x,X,Y):
    n = len(X)
    for i in range(n-1):
        if X[i]<= x <= X[i+1]:
            a = Y[i+1] - Y[i]
            b = (X[i+1] - X[i]).days
            return Y[i] + (x - X[i]).days*(a/b)
    return "No es posible interpolar"


def InterestRateSwap(M, n, fix_rate, flotante,calen_flotante, descuentos, calen_desc,
                     plazos, tau,  payer = False, act_360 = True):
    """
    M: Nocinal
    n: # de cupones
    fix_rate: Tasa fija de la pata larga del IRS
    flotante: Curva de tasas de mercado (la que encuentras en BANXICO) de la pata corta
    calen_flotante: Las fechas dadas por la curva de flotante
    descuentos: Curca de tasas de descuento de mercado para valuar el IRS
    calen_desc: Las fechas dadas por la curva de descuentos
    plazos : Lista de plazo del i -ésimo cupón (habiles)
    tau: Plazo en dias del i-ésimo cupón. (naturales)
    payer: Dummy; 1 si paga fija, 0 si paga flotante
    act_360: Dummy; tipo de convención 1 si act/360, 0 si act/365
    """
    today = datetime.now()
    today = datetime(today.year, today.month, today.day)
    spot = today + timedelta(1)
    
    if act_360:
        conv = 360
    else:
        conv = 365
        
    aux = pd.DataFrame({"Cupon":list(range(1,n+1)), "Tau":tau})
    aux["Desde Spot"] = aux["Tau"].cumsum()
    aux["Fecha"] = aux["Desde Spot"].apply(lambda x: timedelta(x) + spot)
    
    # aux2 = pd.DateFrame({"Cal Flotante":calen_flotante, "Flotante":flotante})
    # aux3 = pd.DataFrame({"Cal Descuento": calen_desc, "Descuentos": descuentos})
    
    df = pd.DataFrame({"Cupon":list(range(1,n+1)), "Plazo": plazos, "Tau": tau})
            
    df["Fechas"] =  aux["Fecha"]
    df["Flotante"] = df["Fechas"].apply(InterpolacionLineal, args=(calen_flotante, flotante))
    df["Descuentos"]=df["Fechas"].apply(InterpolacionLineal, args=(calen_desc, descuentos))
    
    df["Sumando"] = ((df["Flotante"]-fix_rate)*df["Plazo"]/conv) / (1 + (df["Descuentos"]*df["Tau"]/conv))
    
    return M*((-1)**payer)*df["Sumando"].sum()
