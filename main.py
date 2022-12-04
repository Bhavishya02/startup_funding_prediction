import pandas as pd
df = pd.read_csv('startup_funding.csv')
print(df)
inputs = df.drop('AmountUSD', axis='columns')
targets = df['AmountUSD']


from sklearn.preprocessing import LabelEncoder
le_Date = LabelEncoder()
le_Remarks = LabelEncoder()
le_StartupName = LabelEncoder()
le_CityLocation = LabelEncoder()
le_InvestmentType = LabelEncoder()
le_InvestorsName= LabelEncoder()
le_SubVertical = LabelEncoder()
le_IndustryVertical = LabelEncoder()




from sklearn import tree
inputs['Date_n'] = le_Date.fit_transform(inputs['Date'])
inputs['Remarks_n'] = le_Remarks.fit_transform(inputs['Remarks'])
inputs['IndustryVertical_n'] = le_Date.fit_transform(inputs['IndustryVertical'])
inputs['SubVertical_n'] = le_Date.fit_transform(inputs['SubVertical'])
inputs['StartupName_n'] = le_Date.fit_transform(inputs['StartupName'])
inputs['CityLocation_n'] = le_Date.fit_transform(inputs['CityLocation'])
inputs['InvestorsName_n'] = le_Date.fit_transform(inputs['InvestorsName'])
inputs['InvestmentType_n'] = le_Date.fit_transform(inputs['InvestmentType'])
inputs.head()
inputs_n = inputs.drop(['Date', 'Remarks', 'IndustryVertical', 'SubVertical', 'StartupName','CityLocation','InvestorsName','InvestmentType'], axis='columns')

model = tree.DecisionTreeClassifier()
model.fit(inputs_n, targets)
