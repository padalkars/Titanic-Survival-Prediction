#Weight of Evidence(WOE) and Information Value(IV)
'''
Working on the titanic data set.
One can modify the function parameters as per the requirements and the data set one is working on.
'''

#This function will divide the set of values into bins
def get_bins(var, data = train, bins = 10):
    #Numeric Variables
    if(data[var].dtype == int or data[var].dtype == float):
        bins = pd.cut(data[var], bins=bins) #no. of bins can be changed
        return (bins)
    
    #If the variable is categorical each category acts like a bin
    if(data[var].dtype == object):
        categories = list(np.unique(data[var]))
        return (categories)
    return None

#Avoiding
'''
Log(0/1)
Log(1/0)
'''
def woe_iv_sanity(goods, bads, total_goods, total_bads):
    if(goods == 0):
        goods = ((goods + 0.5)/total_goods)
    if(bads == 0):
        bads = ((bads + 0.5)/total_bads)
    
    return (goods, bads)

def get_group_wise_stats(data, total_survived, total_died, target = 'Survived'):
    #Frequency, %goods, %bads
    frequency = data.shape[0]
    no_goods = data.loc[data[target]==1, 'Survived'].shape[0]
    no_bads = frequency - no_goods
    
    #Handling the 0 values
    no_goods, no_bads = woe_iv_sanity(no_goods, no_bads, total_survived, total_died)
    
    per_goods, per_bads = list(map(lambda x: x*100/frequency,\
                                   [no_goods, no_bads]))
    
    return (frequency, per_goods, per_bads)

def WOE_IV(var, target= 'Survived', data=train):
    #Number of survived and died
    total_survived = data.loc[data[target]==1, target].shape[0]
    total_died = data.shape[0] - total_survived
    
    #Obtain the bins
    bins = get_bins(var, data)
    
    #Create a WOE_IV data frame
    WOE_IV_df = pd.DataFrame()
    bins, woe_s, iv_s = [], [], []
    
    if(data[var].dtype == object):
        bin_grp = data.groupby(var)
        for grp, df in bin_grp:
            #Calculate the number of goods and bads
            frequency, per_survived, per_died = get_group_wise_stats(df, total_survived, total_died)
            
            woe_s.append(np.log(per_survived/per_died))
            iv_s.append(per_survived - per_died)
        
        WOE_IV_df[var + '_bins'] = list(bin_grp.groups)
        WOE_IV_df['WOE'] = woe_s
        WOE_IV_df['IV'] = iv_s
    return WOE_IV_df
