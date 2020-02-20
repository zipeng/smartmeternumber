#webapi sepcific
from flask import Flask
from flask import jsonify
from fastai.metrics import dice
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
from flask import abort
from flask import request
import operator
from flask_cors import CORS
CORS(app)
#training data specific

lLocation = []
lWrongCharactor=[]
#comment
lRightCharactor=[]
lLeftNeighbour=[]
lRightNeighbour=[]
def my_function(right, wrong):
    global lLocation
    global lWrongCharactor
    global lRightCharactor
    
    global lLeftNeighbour
    global lRightNeighbour

    sRight=str(right)  
 
    sWrong=str(wrong)
    global x
    for i in range( len(sRight) ):
    #print(sRight[i]+sWrong[i])
        if(sRight[i]!=sWrong[i]):
            print(i)
            lLocation.append(i)
            lWrongCharactor.append(sWrong[i])
            lRightCharactor.append(sRight[i])
            if(i>=7):
                lRightNeighbour.append('10')
            else:
                lRightNeighbour.append(sRight[i+1])
            
            if(i==0):
                lLeftNeighbour.append('10')
            else:
                lLeftNeighbour.append(sRight[i-1])
                
            print(sRight[i]+"   "+sWrong[i])
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
df = pd.read_excel('ErrorData.xlsx', sheet_name='Sheet1')
print(df.columns)

def GenerateExcel(lright, lwrong):
    global lLocation
    global lWrongCharactor
    global lRightCharactor
    global lLeftNeighbour
    global lRightNeighbour
    lLocation = []
    lWrongCharactor=[]
    lRightCharactor=[]
    lLeftNeighbour=[]
    lRightNeighbour=[]

    for i in range( len(lright) ):
        my_function(lright[i],lwrong[i])

GenerateExcel(df['Right'],df['Wrong'])

from pandas import DataFrame
dfwrite = DataFrame({'Location': lLocation, 'lWrongCharactor': lWrongCharactor,'lRightCharactor':lRightCharactor,'lLeftNeighbour':lLeftNeighbour,'lRightNeighbour':lRightNeighbour})

dfwrite.to_excel('test.xlsx', sheet_name='sheet1', index=False)

dfx=dfwrite[['lWrongCharactor','lRightNeighbour','lLeftNeighbour']] 



lx=dfx.values.tolist()

#caculate probablity
testdf=dfwrite[['lWrongCharactor','lRightCharactor']]
nAllNum=testdf.count()['lWrongCharactor']
testdf['total']=nAllNum
testdf['count']=testdf.groupby(['lWrongCharactor','lRightCharactor']).transform('count')
testdf['countWrong']=testdf.groupby('lWrongCharactor')['lRightCharactor'].transform('count')
testdf['probablity']=testdf['countWrong']/testdf['total']

#probablity with neighbour
dfNeighbourProb=dfwrite[['lWrongCharactor','lRightNeighbour','lLeftNeighbour']] 
dfNeighbourProb['total']=nAllNum
dfNeighbourProb['countw']=dfNeighbourProb.groupby(['lWrongCharactor','lLeftNeighbour','lRightNeighbour']).transform('count')

NeighbourProb=dfNeighbourProb.drop_duplicates()

def GetProbilityNeighbour(sWrongChr,sLeftNeighbour,sRightNeighbour):
    global NeighbourProb
    global nAllNum
    Filtered=NeighbourProb.loc[(NeighbourProb['lWrongCharactor'] == sWrongChr)&(NeighbourProb['lRightNeighbour'] == sRightNeighbour)&(NeighbourProb['lLeftNeighbour'] == sLeftNeighbour)]
    if(Filtered.empty):
        return GetProbility(sWrongChr)/100
    else:
        return Filtered['countw'].values[0]/nAllNum


testdf.to_excel('test1.xlsx', sheet_name='sheet1', index=False)

probablitydf=testdf[['lWrongCharactor','probablity']]
Ptobdf=probablitydf.drop_duplicates()

#probablity
WrongCharadf=Ptobdf[['lWrongCharactor']]
Probablity=Ptobdf[['probablity']]
lWrongCharactors=WrongCharadf.values.tolist()
lProbablities=Probablity.values.tolist()
listw=sum(lWrongCharactors, [])
listp=sum(lProbablities, [])
ProbablityDic = dict(zip(listw, listp))


def GetProbility(sWrongCharactor):
    global ProbablityDic
    if sWrongCharactor in ProbablityDic.keys():
        return ProbablityDic[sWrongCharactor]
    else:
        return 0.01

def getRightProb(sWrong,sRight):
    global testdf
    global nAllNum
    Filtered=testdf.loc[(testdf['lWrongCharactor']==sWrong)&(testdf['lRightCharactor']==sRight)]
    if(Filtered.empty):
        return GetProbility(sWrong)/100
    else:
        return Filtered['count'].values[0]/nAllNum

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf.fit(lx, lRightCharactor);


#Similarity specific

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


#MeterType and NumberRange

#dfMeterType=pd.read_excel('MeterType.xlsx', sheet_name='Sheet1')
dfNumberRange=pd.read_excel('NumberRange.xlsx', sheet_name='sheet1')

dfMeterType=dfNumberRange[['MeterType']] 

lMeterType=dfMeterType.values.tolist()
lNumberRange=dfNumberRange.values.tolist()

def GetNumberRange(sMeterType):
    global lNumberRange
    for NumberRange in lNumberRange:
        if NumberRange[0]==sMeterType:
            return dict(From=NumberRange[1], To=NumberRange[2])

#dfNumberRange.loc[dfwrite['Location'] == 7]

#based on number range devide meternumber from 14 to 9+5. 


#getMeterType
from operator import itemgetter
def getMeterType(sWrongNumber):
    sWrongType = sWrongNumber[0:9]
    global lNumberRange
    lMeterType=[]
    for i in lNumberRange: 
        sMeterType=i[0]
        
        sSimilar=similar(sMeterType,sWrongType)
       
        possiblenumbers = dict(MeterType=sMeterType, Possibility=sSimilar)
        lMeterType.append(possiblenumbers)
    
    newlist = sorted(lMeterType, key=itemgetter('Possibility'),reverse=True)
    return newlist[0]

#NumberRange

def checkNumberRangeNew2(sWrongNumber,sFrom,sTo):
    lr=[]
    lp=[]
    nFrom=int(sFrom[0])
    nTo=int(sTo[0])
    sWrongChar=sWrongNumber[0]
    sPossibleRight=sWrongNumber
    for i in range(nFrom,nTo+1):
        sPossibleRight=str(i) +sPossibleRight[1:]
        sRChara=sPossibleRight[0] 
        nProbablity=getRightProb(sWrongChar,sRChara)
        if(sRChara!=sWrongNumber[0]):
            lr.append(sPossibleRight)
            lp.append(nProbablity)
    lrd=dict(zip(lr, TransformProb(lp)))
    sorted_d = dict( sorted(lrd.items(), key=operator.itemgetter(1),reverse=True))
    key=list(sorted_d)[0]
    value=sorted_d[key]
    prob=GetProbility(sWrongChar)
    if(int(sWrongChar)<=nTo and int(sWrongChar)>=nFrom):
        realprob=min(value,prob)
    else:
        realprob=max(value,prob)
    return {key:realprob}

def checkNumberRangeNew(sWrongNumber,sFrom,sTo,sLeftN):
    nWrongNumber=int(sWrongNumber[0])
    nRightN=int(sWrongNumber[1])
    nLeftN=int(sLeftN)
    nFrom=int(sFrom[0])
    nTo=int(sTo[0])
    sPossibleRight=sWrongNumber
    
    lr=[]
    lp=[]
    nPossibleRight=int(clf.predict([[nWrongNumber,nLeftN,nRightN]])[0]);
    
    if(nWrongNumber<=nTo and nWrongNumber>=nFrom):
        sWChara=sWrongNumber[0]
        nProbablity=1-GetProbility(sWChara)
        
        #lr.append(sWrongNumber)
        #lp.append(nProbablity)
      
        print(lr)
        print(lp)
        
    
    if(nPossibleRight<=nTo and nPossibleRight>=nFrom):
        sPossibleRight=str(nPossibleRight) +sWrongNumber[1:]
        
        sWChara=sPossibleRight[0]
        nProbablity=GetProbility(sWChara)
        if(sWChara!=sWrongNumber[0]):
            lr.append(sPossibleRight)
            lp.append(nProbablity)
            
            print('return1')
            print(lr)
            print(lp)
        lrd=dict(zip(lr, lp))
        return lrd
      
    else:
        for i in range(nFrom,nTo):
            sPossibleRight=str(i) +sPossibleRight[1:]
        
            sWChara=sPossibleRight[0]
            nProbablity=1.0/(len(range(nFrom,nTo))+1)
            if(sWChara!=sWrongNumber[0]):
                lr.append(sPossibleRight)
                lp.append(nProbablity)
            
        sPossibleRight=str(nTo) +sPossibleRight[1:]    
        sWChara=sPossibleRight[0]
        nProbablity=1.0/(len(range(nFrom,nTo))+1)
        if(sWChara!=sWrongNumber[0]):
            lr.append(sPossibleRight)
            lp.append(nProbablity)
        lrd=dict(zip(lr, lp))
        print('return2')
        return lrd

def checkNumberRange(sWrongNumber,sFrom,sTo,sLeftN):
    nWrongNumber=int(sWrongNumber[0])
    nRightN=int(sWrongNumber[1])
    nLeftN=int(sLeftN)
    nFrom=int(sFrom[0])
    nTo=int(sTo[0])
    sPossibleRight=sWrongNumber
    
    lr=[]
    nPossibleRight=int(clf.predict([[nWrongNumber,nLeftN,nRightN]])[0]);
    
    if(nWrongNumber<=nTo and nWrongNumber>=nFrom):
        
        lr.append(sWrongNumber)
        
        return lr
    
    if(nPossibleRight<=nTo and nPossibleRight>=nFrom):
        sPossibleRight=str(nWrongNumber) +sWrongNumber[1:]
        lr.append(sPossibleRight)
        return lr
      
    else:
        for i in range(nFrom,nTo):
           sPossibleRight=str(i) +sPossibleRight[1:]
           lr.append(sPossibleRight)
        sPossibleRight=str(nTo) +sPossibleRight[1:]    
        lr.append(sPossibleRight)
        return lr
        
#SimpleNumbers

def TransformProb(lprob):
    sum=0
    lreturn=[]
    for i in lprob:
        sum=sum+i
    for i in lprob:
        lreturn.append(i/sum)
    return lreturn   

def SimpleNumberNew(sWrongNumber):
    lWrongNumber=list(sWrongNumber)
    lPossibleNum=lWrongNumber
    lArray=[]
    lPossibleRightNumber=[]
    lPorbility=[]
    print()
    for i in range(len(sWrongNumber)):
        lPossibleNum=list(sWrongNumber)
        if (i==0):
            lpredictarray=[]
        elif(i==len(sWrongNumber)-1):
            lpredictarray=[]
            nLeftNeighbour=lWrongNumber[i-1]
            nWrong=lWrongNumber[i]
            nRightNeighbour=10
            lpredictarray.append(nWrong)
            lpredictarray.append(nLeftNeighbour)
            lpredictarray.append(nRightNeighbour)
            #lArray.append(lpredictarray)
            possible=clf.predict([lpredictarray])[0]
            nProbablity=GetProbility(str(nWrong))
            nNewProb=GetProbilityNeighbour(str(nWrong),str(nLeftNeighbour),str(nRightNeighbour))
            if(possible!=nWrong):
                lPossibleNum[i]=possible
                sPossibleNum=''.join(lPossibleNum)
                lPossibleRightNumber.append(sPossibleNum)
                lPorbility.append(nNewProb)
            #nPossibleRight=clf.predict([[nWrong,nLeftNeighbour,nRightNeighbour]])
        else:
            lpredictarray=[]
            nLeftNeighbour=lWrongNumber[i-1]
            nWrong=lWrongNumber[i]
            nRightNeighbour=lWrongNumber[i+1]
            lpredictarray.append(nWrong)
            lpredictarray.append(nLeftNeighbour)
            lpredictarray.append(nRightNeighbour)
            #lArray.append(lpredictarray)
            possible=clf.predict([lpredictarray])[0]
            print(nWrong)
            nProbablity=GetProbility(str(nWrong))
            nNewProb=GetProbilityNeighbour(str(nWrong),str(nLeftNeighbour),str(nRightNeighbour))
            if(possible!=nWrong):
                lPossibleNum[i]=possible
                sPossibleNum=''.join(lPossibleNum)
                lPossibleRightNumber.append(sPossibleNum)
                lPorbility.append(nNewProb)
                
    lrd=dict(zip(lPossibleRightNumber, TransformProb(lPorbility)))            
    return  lrd

def SimpleNumber(sWrongNumber):
    lWrongNumber=list(sWrongNumber)
    lPossibleNum=lWrongNumber
    lArray=[]
    lPossibleRightNumber=[]
    print()
    for i in range(len(sWrongNumber)):
        lPossibleNum=list(sWrongNumber)
        if (i==0):
            lpredictarray=[]
        elif(i==len(sWrongNumber)-1):
            lpredictarray=[]
            nLeftNeighbour=lWrongNumber[i-1]
            nWrong=lWrongNumber[i]
            nRightNeighbour=10
            lpredictarray.append(nWrong)
            lpredictarray.append(nLeftNeighbour)
            lpredictarray.append(nRightNeighbour)
            #lArray.append(lpredictarray)
            possible=clf.predict([lpredictarray])[0]
            if(possible!=nWrong):
                lPossibleNum[i]=possible
                sPossibleNum=''.join(lPossibleNum)
                lPossibleRightNumber.append(sPossibleNum)
            #nPossibleRight=clf.predict([[nWrong,nLeftNeighbour,nRightNeighbour]])
        else:
            lpredictarray=[]
            nLeftNeighbour=lWrongNumber[i-1]
            nWrong=lWrongNumber[i]
            nRightNeighbour=lWrongNumber[i+1]
            lpredictarray.append(nWrong)
            lpredictarray.append(nLeftNeighbour)
            lpredictarray.append(nRightNeighbour)
            #lArray.append(lpredictarray)
            possible=clf.predict([lpredictarray])[0]
            if(possible!=nWrong):
                lPossibleNum[i]=possible
                sPossibleNum=''.join(lPossibleNum)
                lPossibleRightNumber.append(sPossibleNum)
    return  lPossibleRightNumber     

#generate result

def MergeRangeSimmple(lRange,lSimple):
    lResult=[]
    for i in range(len(lRange)):
        
        lResult.append(lRange[i]) 
        sR=lRange[i][0]
        for j in range(len(lSimple)):
            sResult=sR+lSimple[j][1:]
            lResult.append(sResult)
    return lResult  

def getPossibleMeterNumber(dMeterType,lPossibleNumbers,sSubNumber):
    lPossibleMeterNumber=[]
    sMeterType=dMeterType['MeterType']
    sSimiraty=dMeterType['Possibility']
    if(int(sSimiraty)<1):
        sPossibleMeter=sMeterType+sSubNumber
        #lPossibleMeterNumber.append(sPossibleMeter)
    for i in range(len(lPossibleNumbers)):
        sPossibleMeter=sMeterType+lPossibleNumbers[i]
        if(int(sSimiraty)>=1 and lPossibleNumbers[i]==sSubNumber):
            sPossibleMeter=sMeterType+sSubNumber
        else:
            lPossibleMeterNumber.append(sPossibleMeter)    
    return  lPossibleMeterNumber            

def MergePossibleMeterNumber(dMeterType, dSorted,sSubnumber):
    lPossibleMeterNumber=[]
    lPorbility=[]
    sMeterType=dMeterType['MeterType']
    nSimiraty=dMeterType['Possibility']
    if(int(nSimiraty)<1):
        sPossibleNumber=sMeterType+sSubnumber
        lPossibleMeterNumber.append(sPossibleNumber)
        lPorbility.append(nSimiraty)
    for key, value in dSorted.items(): 
        sPossibleNumber=sMeterType+key
        lPossibleMeterNumber.append(sPossibleNumber)
        lPorbility.append(value)
    lrd=dict(zip(lPossibleMeterNumber, lPorbility))            
    return  lrd  

#webapi

@app.route("/")
def hello():
        return "Hello World!"
    


@app.route('/predict/api/predictnumbers', methods=['POST'])
def create_task():
    if not request.json or not 'wrongNumber' in request.json:
        abort(400)
    
    sWrongNumber=str(request.json['wrongNumber'])
    sSubNumber=sWrongNumber[9:]
    
    # check meter type from wrong number
    dMeterType=getMeterType(sWrongNumber)
    sMeterType=dMeterType['MeterType']
    sSimiraty=dMeterType['Possibility']
    
    # get number Range from wrong number
    dNumberRange=GetNumberRange(sMeterType)
    sFrom=dNumberRange['From']
    sTo=dNumberRange['To']
    sLeftN=sWrongNumber[8]
    
    
    # check number Range from wrong number
    lPossibleRange=checkNumberRange(sSubNumber,sFrom[9:],sTo[9:],sLeftN)
    dPossibleRange=checkNumberRangeNew2(sSubNumber,sFrom[9:],sTo[9:])
    # Check simple wrong number
    
    lSimpleNumber=SimpleNumber(sSubNumber)
    dSimpleNumber=SimpleNumberNew(sSubNumber)
    
    dPossibleNumbers= {**dPossibleRange , **dSimpleNumber}
    sorted_d = dict( sorted(dPossibleNumbers.items(), key=operator.itemgetter(1),reverse=True))
    
    dPossibleMeters=MergePossibleMeterNumber(dMeterType,sorted_d,sSubNumber)
    
    dSortPossibleMeters=dict( sorted(dPossibleMeters.items(), key=operator.itemgetter(1),reverse=True))
    
    lPossibleNumbers=MergeRangeSimmple(lPossibleRange,lSimpleNumber)
    
    lPossibleMeterNumbers=getPossibleMeterNumber(dMeterType,lPossibleNumbers,sSubNumber)
    

    lreturn=[]
    
    for key in dSortPossibleMeters:
        tmpdic={'PossibleNumber': key,'Conficence':dSortPossibleMeters[key]};
        lreturn.append(tmpdic)
    
    
    
    #possiblenumbers = dict(PossibleNumber =  request.json['wrongNumber'], Possibility  = "80%")
    
    
    
    
    #pRoposedNumbers.append(lPossibleMeterNumbers)

    return jsonify({'ProposedNumbers': lreturn}),201
   

if __name__ == "__main__":
    app.run()