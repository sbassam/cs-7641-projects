import pandas as pd
import data
import traitlets.utils.bunch



def process_abalone():
    df = pd.read_csv('data/abalone.data', names = ["Sex", "Length", "Diameter", "Height",
                                                   "Whole weight", "Shucked weight", "Viscera weight",
                                                   "Shell weight", "Rings"])
    df = df[(df["Height"]!=1.13) & (df['Height']!=0.515)]

    # deal with categorical data
    df.loc[df.Sex == 'M', 'Male'] = 1.
    df.loc[df.Sex == 'F', 'Female'] = 1.
    df.loc[df.Sex == 'I', 'Infant'] = 1.
    df.fillna(0, inplace=True)

    return traitlets.Bunch(data = df[['Male', 'Female', 'Infant', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                             'Viscera weight', 'Shell weight']].values,
                           target = df[['Rings']].values,
                           target_names = df["Rings"].unique(),
                           DESCR = 'abalone dataset...',
                           feature_names = ['Male', 'Female', 'Infant', "Length", "Diameter", "Height",
                                                   "Whole weight", "Shucked weight", "Viscera weight",
                                                   "Shell weight"],
                           )


