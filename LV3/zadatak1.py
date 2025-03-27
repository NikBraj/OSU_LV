import pandas as pd 

data = pd.read_csv('data_C02_emission.csv')

# Zadatak pod a)
length = len(data['Make'])
print(f'DataFrame ima {length} mjerenja')

for col in data.columns:
    print(f"{col} has a type of {data[col].dtype}")

data['Vehicle Class'] = data['Vehicle Class'].astype('category')

print(f"Redovi s izostalim vrijednostima: {data.isnull().sum()}")
print(f"Duplicirane vrijednosti: {data.duplicated().sum()}")

print('-------------------------------------------------------------------------------')

# Zadatak pod b)
least_consuming = data.nsmallest(3, 'Fuel Consumption City (L/100km)')
most_consuming = data.nlargest(3, 'Fuel Consumption City (L/100km)')

print('Most consuming: ')
print(most_consuming[['Make', 'Model', 'Fuel Consumption City (L/100km)']])
print('Least consuming: ')
print(least_consuming[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

print('-------------------------------------------------------------------------------')

# Zadatak pod c) 
selected_data = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
length = len(selected_data['Make'])
print(f"Postoji {length} vozila koje imaju motor izmedu 2.5 i 3.5 L")

print(f"Prosjecni C02 ovih vozila jest: {selected_data['CO2 Emissions (g/km)'].mean()} g/km")

print('-------------------------------------------------------------------------------')

# Zadatak pod d)
selected_data = data[(data['Make'] == 'Audi')]
length = len(selected_data['Make'])
print(f"U mjerenjima ima {length} mjerenja koja se odnose na marku Audi")

selected_data = selected_data[(selected_data['Cylinders'] == 4)]
print(f"Prosjecni CO2 4 cilindrasa marke Audi je {selected_data['CO2 Emissions (g/km)'].mean()} g/km")

print('-------------------------------------------------------------------------------')

# Zadatak pod e)
cylinder_counts = data['Cylinders'].value_counts().sort_index()
print(cylinder_counts)

cylinder_emissions = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print("Cylinder emissions: ")
print(cylinder_emissions)

print('-------------------------------------------------------------------------------')

# Zadatak pod f)
diesels = data[(data['Fuel Type'] == 'D')]
petrols = data[(data['Fuel Type'] == 'Z')]

print(f"Dizeli:\nProsjecno: {diesels['Fuel Consumption City (L/100km)'].mean()} - Medijalno: {diesels['Fuel Consumption City (L/100km)'].median()}")
print(f"Benzinci:\nProsjecno: {petrols['Fuel Consumption City (L/100km)'].mean()} - Medijalno: {petrols['Fuel Consumption City (L/100km)'].median()}")

print('-------------------------------------------------------------------------------')

# Zadatak pod g)
four_cylinder_diesels = diesels[(diesels['Cylinders'] == 4)]
print(f"4 cilindricni dizel koji najvise goriva trosi u gradu jest:\n{four_cylinder_diesels.nlargest(1, 'Fuel Consumption City (L/100km)')}")

print('-------------------------------------------------------------------------------')

# Zadatak pod h)
manuals = data[(data['Transmission'].str[0] == 'M')]
length = len(manuals['Make'])
print(f"Postoji {length} vozila s rucnim mjenjacem")

print('-------------------------------------------------------------------------------')

#Zadatak pod i)
print(data.corr(numeric_only=True))

print('-------------------------------------------------------------------------------')

'''
Komentar na zadnji zadatak:
Veličine u podatkovnom skupu pokazuju prilično visoku međusobnu korelaciju. Na primjer, broj cilindara i obujam motora imaju 
korelaciju od oko 0.9, što ukazuje na snažnu povezanost. Također, potrošnja goriva izražena u L/100 km ima korelaciju od oko 0.8 
s tim veličinama, što je očekivano jer su svi ti parametri povezani s veličinom i snagom vozila.
S druge strane, potrošnja izražena u MPG (miles per gallon) pokazuje jaku negativnu korelaciju. Razlog tome je što je ta mjera 
obrnutog značenja — što je broj MPG veći, to vozilo manje troši goriva. Primjerice, automobil koji troši 15 MPG troši više goriva 
od onoga koji troši 55 MPG. Zbog toga dolazi do negativne korelacije: kako rastu vrijednosti ostalih parametara 
(npr. broj cilindara), vrijednost MPG opada.
Negativna korelacija blizu -1 označava gotovo savršeno obrnutu proporcionalnost, dok korelacija blizu 1 označava izravnu proporcionalnost.
Vrijednosti blizu 0 znače da između varijabli nema značajne povezanosti.
'''