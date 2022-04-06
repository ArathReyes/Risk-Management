# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:21:15 2022

@author: Arath Reyes
"""


import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay





def InterpolacionLineal(x,X,Y):
    n = len(X)
    for i in range(n-1):
        if X[i]<= x <= X[i+1]:
            a = Y[i+1] - Y[i]
            b = (X[i+1] - X[i]).days
            return Y[i] + (x - X[i]).days*(a/b)
    return "No es posible interpolar"

class InterestRateSwap:
    
    def __init__(self):
        self.precio = None
        self.n_cupones = None
        self.nocional = None
        self.posicion = None
        self.convencion = None
        self.fija = None
        self.flotante = None
        self.descuentos = None
        self.summary = None
        
    def compute(self, M, n, fix_rate, flotante,calen_flotante, descuentos, calen_desc, 
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
                
        df = pd.DataFrame({"Cupon":list(range(1,n+1)), "Plazo": plazos, "Tau": tau})
                
        df["Fechas"] =  aux["Fecha"]
        df["Flotante"] = df["Fechas"].apply(InterpolacionLineal, args=(calen_flotante, flotante))
        df["Descuentos"]=df["Fechas"].apply(InterpolacionLineal, args=(calen_desc, descuentos))
        
        # Calculo de tasas forward
        desc_1_dia=np.exp(-flotante[0]*((spot-today).days)/conv)
        auxl= np.array(np.zeros([len(df['Flotante'])]))
        auxl[0]=desc_1_dia*(1+df['Flotante'][0]*df['Tau'][0])**(-1)
        for i in range(1,len(df['Flotante'])):
            auxl[i]=(1-df['Flotante'][i]*sum(df['Tau'][:i]*auxl[:i]))/(1+df['Flotante'][i]*df['Tau'][i])
        df['Cupon_Flotante'] = auxl
        df['Forward_Flotante'] = (df['Cupon_Flotante'] - df['Cupon_Flotante'].shift(1))/ (df['Tau']*df['Cupon_Flotante'].shift(1))
        df['Forward_Flotante'][0] = df['Flotante'][0]
        df["Sumando"] = ((df["Forward_Flotante"]-fix_rate)*df["Plazo"]/conv) / (1 + (df["Descuentos"]*df["Tau"]/conv))
        
        self.precio =  M*((-1)**payer)*df["Sumando"].sum()
        self.n_cupones = n
        self.nocional = M
        self.posicion = "Payer" if payer else "Receiver"
        self.convencion = conv
        self.fija = fix_rate
        self.flotante = df[["Fechas","Forward_Flotante"]]
        self.descuentos = df[["Fechas", "Descuentos"]]
        self.summary = df
        
        return
    
    def plots(self):
        Bool = True
        while Bool:
            print("\n")
            choice = input("Si deseas ver las tasas de descuento teclea 'Descuentos', si deseas\
                                ver las tasas flotantes teclea 'Tasas' o si deseas salir escribe '0':\n")
            if choice == 'Descuentos' or choice =='Tasas' or choice == '0':
                Bool = False
            else:
                print('\nTu elección no es válida')
        if choice == 'Descuentos':
            sns.set_style('darkgrid')
            sns.set_palette('tab10')
            plt.figure(figsize = (12,8))
            ax = sns.lineplot(x = self.summary['Fechas'], y = self.summary['Descuentos'])
            ax.set_title("Curva de Descuentos",fontsize = '25')
            plt.show()
            
            return
        elif choice == 'Tasas':
            sns.set_style('darkgrid')
            sns.set_palette('tab10')
            plt.figure(figsize = (12,8))
            ax = sns.lineplot(x = self.summary['Fechas'], y = self.summary['Forward_Flotante'], color = 'red')
            ax.set_title("Curva de Tasas Forward Flotantes",fontsize = '25')
            plt.show()
            
            return
        else:
            return
        
        

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
