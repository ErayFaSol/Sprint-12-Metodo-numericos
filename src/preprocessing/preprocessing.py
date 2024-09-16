import pandas as pd
from datetime import datetime

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    #Eliminar ffilas duplicadas
    df = df.drop_duplicates()
    # Imputar valores nulos con la moda para las columnas categoricas
    df = df[df['Price'] > 0]
    df['VehicleType'] = df['VehicleType'].fillna(df['VehicleType'].mode()[0])
    df['Gearbox'] = df['Gearbox'].fillna(df['Gearbox'].mode()[0])
    df['Model'] = df['Model'].fillna(df['Model'].mode()[0])
    df['FuelType'] = df['FuelType'].fillna(df['FuelType'].mode()[0])
    df['NotRepaired'] = df['NotRepaired'].fillna('no')
    # Realizar otras limpiezas...
    # Eliminar filas donde el precio es 0
    df = df[df['Price'] > 0]
    # Definir el rango de años de registro válidos
    current_year = datetime.now().year
    df = df[(df['RegistrationYear'] >= 1900) & (df['RegistrationYear'] <= current_year)]
    
    # Eliminar filas donde la potencia es 0
    df = df[df['Power'] > 0]
    # Convertir columnas de fechas a formato datetime
    df['DateCrawled'] = pd.to_datetime(df['DateCrawled'], format='%d/%m/%Y %H:%M')
    df['DateCreated'] = pd.to_datetime(df['DateCreated'], format='%d/%m/%Y %H:%M')
    df['LastSeen'] = pd.to_datetime(df['LastSeen'], format='%d/%m/%Y %H:%M')

    # Eliminar columnas irrelevantes
    df = df.drop(columns=['NumberOfPictures'])
    
    return df
