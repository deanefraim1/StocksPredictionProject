import pandas as pd
import yfinance as yf

firstStockSymbol = "AAPL"
secondStockSymbol = "MSFT"

startDate = "2020-01-01"
endDate = "2021-01-01"

TemplateInstanceFilePath = 'RDDL/Instance0_Template.rddl'
InstanceFilePath = 'RDDL/Instance0.rddl'

RddlTimesVar = "{$timesList}"
RddlStock1PricesVar = "{$stock1PricesList}"
RddlStock2PricesVar = "{$stock2PricesList}"
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

def CreatePricesListString(fileData: pd.DataFrame): # create a string of {price0, price1, price2, ..., priceN}
    open_prices = fileData['Open']
    close_prices = fileData['Close']
    avg_prices = ((open_prices + close_prices) / 2).round(3)
    concatenated_string = '{' + ', '.join(map(str, avg_prices)) + '}'
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
    rddlTemplate = rddlTemplate.replace(RddlStock1PricesVar, CreatePricesListString(firstStockData))
    rddlTemplate = rddlTemplate.replace(RddlStock2PricesVar, CreatePricesListString(secondStockData))
    rddlTemplate = rddlTemplate.replace(RddlNextTimeBoolsVar, CreateNextTimeBoolsString(firstStockData))
    rddlTemplate = rddlTemplate.replace(RddlTimeTailVar, CreateTimeTailstring(firstStockData))

    rddlFile = open(instanceName, 'w')
    rddlFile.write(rddlTemplate)
    rddlFile.close()


firstStockData = yf.download(firstStockSymbol, start=startDate, end=endDate)
secondStockData = yf.download(secondStockSymbol, start=startDate, end=endDate)

GenerateRddlFromStockData(TemplateInstanceFilePath, firstStockData, secondStockData, InstanceFilePath)