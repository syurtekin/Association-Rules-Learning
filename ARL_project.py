import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


df = retail_data_prep(df)

# GOREV 2
df_ger = df[df['Country'] == "Germany"]

# bir faturadaki ürünler ve miktarları
df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]


df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]



def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


ger_inv_pro_df = create_invoice_product_df(df_ger)

ger_inv_pro_df = create_invoice_product_df(df_ger, id=True)


# GOREV 3
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_ger, 21987) #PACK OF 6 SKULL PAPER CUPS
check_id(df_ger, 23235) # STORAGE TIN VINTAGE LEAF
check_id(df_ger, 22747) #POPPY'S PLAYHOUSE BATHROOM

############################################
# Birliktelik Kurallarının Çıkarılması
############################################
# antecedents: önceki ürün, consequents: sonraki ürün ,antecedents support : önceki ürünün tek başına olasılığını verir.
# consequents support: sonraki ürünün tek başına gözükme olasılığını verir
# support : iki ürünün birlikte görülme olasılığını ifade eder.
# confidence: x alındığında y nin alınma olasılığı
# lift: x alındığında y nın kaç kat alınma olasılığı

frequent_itemsets = apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(20)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head(100)

rules.sort_values("lift", ascending=False).head(100)


# GOREV 4 : Sepetteki kullanıcılar için ürün önerisi yapınız.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]
#
arl_recommender(rules, 21987,2) #21086, 21989
check_id(df_ger, 21086) # SET/6 RED SPOTTY PAPER CUPS
check_id(df_ger, 21989) # PACK OF 20 SKULL PAPER NAPKINS
arl_recommender(rules, 23235, 2) #23244 # 23240
check_id(df_ger, 23240) # SET OF 4 KNICK KNACK TINS DOILEY
check_id(df_ger, 23244) # ROUND STORAGE TIN VINTAGE LEAF
arl_recommender(rules, 22747,1)  #22746
check_id(df_ger, 22746) # POPPY'S PLAYHOUSE LIVINGROOM

# sonuçlar anlamlı, birbirine yakın, benzer ürünler önerdi.
