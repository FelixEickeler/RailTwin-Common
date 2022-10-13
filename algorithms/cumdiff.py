# 05.04.22----------------------------------------------------------------------------------------------------------------------
#  created by: Felix Eickeler 
#              felix.eickeler@tum.de       
# ----------------------------------------------------------------------------------------------------------------
#
#
# see https://stackoverflow.com/questions/57261855/subtract-current-column-value-from-the-previous-column-of-the-same-row-in-pandas

def cumdiff(df, axis):
    if axis==1 :
        for x, y in enumerate(df.columns):
            if x == 0:
                df[y] = df[y]
            else:
                df[y] = df[y] - df[df.columns[x - 1]]
        return df
    else :
        for x, y in enumerate(df.index):
            if x == 0:
                df[y] = df[y]
            else:
                df[y] = df[y] - df[df.columns[x - 1]]
        return df