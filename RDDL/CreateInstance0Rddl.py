import csv
import pandas as pd

firstStockDataFilePath = 'StocksData/S&P500_Data.csv'
secondStockDataFilePath = 'StocksData/S&P500_Data copy.csv'
OpenPricesColumnName = 'Open'
ClosePricesColumnName = 'Close/Last'
TemplateInstanceFilePath = 'RDDL/Instance0_Template.rddl'
InstanceFilePath = 'RDDL/Instance0.rddl'

RddlTimesVar = "{$timesList}"
RddlStock1OpenPricesVar = "{$stock1OpensList}"
RddlStock1ClosePricesVar = "{$stock1ClosesList}"
RddlStock2OpenPricesVar = "{$stock2OpensList}"
RddlStock2ClosePricesVar = "{$stock2ClosesList}"
RddlNextTimeBoolsVar = "{$nextTimeBools}"
RddlTimeTailVar = "{$timeTailString}"



def ReadFile(file):
    return pd.read_csv(file)

def GetNumberOfRows(fileData: pd.DataFrame):
    return len(fileData) - 1 # -1 because the first row is the header

def CreateTimesListString(fileData: pd.DataFrame): # create a string of {t0, t1, t2, ..., tN}
    NumberOfRows = GetNumberOfRows(fileData)
    timesListString = "{"
    for i in range(NumberOfRows):
        timesListString += "t" + str(i) + ", "
    timesListString = timesListString[:-2] + "}"
    return timesListString

def CreateOpenPricesListString(fileData: pd.DataFrame): # create a string of {openPrice0, openPrice1, openPrice2, ..., openPriceN}
    open_prices = fileData[OpenPricesColumnName]
    concatenated_string = '{' + ', '.join(map(str, open_prices)) + '}'
    return concatenated_string

def CreateClosePricesListString(fileData: pd.DataFrame): # create a string of {closePrice0, closePrice1, closePrice2, ..., closePriceN}
    close_prices = fileData[ClosePricesColumnName]
    concatenated_string = '{' + ', '.join(map(str, close_prices)) + '}'
    return concatenated_string

def CreateNextTimeBoolsString(file: pd.DataFrame): # create a string of NEXT(t0, t1) newLine NEXT(t1, t2) newLine ... NEXT(tN-1, tN)
    NumberOfRows = GetNumberOfRows(file)
    nextTimeBoolsString = ""
    for i in range(NumberOfRows - 1):
        nextTimeBoolsString += "NEXT(t" + str(i) + ", t" + str(i + 1) + ")" + "           = true;\n        "
    return nextTimeBoolsString

def CreateTimeTailstring(file: pd.DataFrame): # create a string of TIME-TAIL(tN)  
    NumberOfRows = GetNumberOfRows(file)
    return "TIME-TAIL(t" + str(NumberOfRows) + ")" + "           = true;"

def GenerateRddlFromStockData(rddlTemplateFilePath, firstStockData: pd.DataFrame, secondStockData: pd.DataFrame, instanceName): # replace {variables} with the actual values
    rddlTemplateFile = open(rddlTemplateFilePath, 'r')
    rddlTemplate = rddlTemplateFile.read()
    rddlTemplateFile.close()
    rddlTemplate = rddlTemplate.replace(RddlTimesVar, CreateTimesListString(firstStockData))
    rddlTemplate = rddlTemplate.replace(RddlStock1OpenPricesVar, CreateOpenPricesListString(firstStockData))
    rddlTemplate = rddlTemplate.replace(RddlStock2OpenPricesVar, CreateOpenPricesListString(secondStockData))
    rddlTemplate = rddlTemplate.replace(RddlStock1ClosePricesVar, CreateClosePricesListString(firstStockData))
    rddlTemplate = rddlTemplate.replace(RddlStock2ClosePricesVar, CreateClosePricesListString(secondStockData))
    rddlTemplate = rddlTemplate.replace(RddlNextTimeBoolsVar, CreateNextTimeBoolsString(firstStockData))
    rddlTemplate = rddlTemplate.replace(RddlTimeTailVar, CreateTimeTailstring(firstStockData))

    rddlFile = open(instanceName, 'w')
    rddlFile.write(rddlTemplate)
    rddlFile.close()


firstStockFileData = ReadFile(firstStockDataFilePath)
secondStockFileData = ReadFile(firstStockDataFilePath)

GenerateRddlFromStockData(TemplateInstanceFilePath, firstStockFileData, secondStockFileData, InstanceFilePath)