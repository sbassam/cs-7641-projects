import pandas as pd
import data
import traitlets.utils.bunch


def process_abalone():
    df = pd.read_csv('data/abalone.data', names=["Sex", "Length", "Diameter", "Height",
                                                 "Whole weight", "Shucked weight", "Viscera weight",
                                                 "Shell weight", "Rings"])
    df = df[(df["Height"] != 1.13) & (df['Height'] != 0.515)]

    # deal with categorical data
    df.loc[df.Sex == 'M', 'Male'] = 1.
    df.loc[df.Sex == 'F', 'Female'] = 1.
    df.loc[df.Sex == 'I', 'Infant'] = 1.
    df.fillna(0, inplace=True)

    return traitlets.Bunch(
        data=df[['Male', 'Female', 'Infant', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                 'Viscera weight', 'Shell weight']].values,
        target=df[['Rings']].values,
        target_names=df["Rings"].unique(),
        DESCR='abalone dataset...',
        feature_names=['Male', 'Female', 'Infant', "Length", "Diameter", "Height",
                       "Whole weight", "Shucked weight", "Viscera weight",
                       "Shell weight"],
    )


def process_red_wine_quality():
    df = pd.read_csv('data/winequality-red.csv', sep=';')
    return traitlets.Bunch(
        data=df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']].values,
        target=df[['quality']].values,
        target_names=df["quality"].unique(),
        DESCR='red wine quality dataset...',
        feature_names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol'],
    )


def process_white_wine_quality():
    df = pd.read_csv('data/winequality-white.csv', sep=';')
    return traitlets.Bunch(
        data=df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']].values,
        target=df[['quality']].values,
        target_names=df["quality"].unique(),
        DESCR='white wine quality dataset...',
        feature_names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol'],
    )


def process_wine_quality():
    df1 = pd.read_csv('data/winequality-red.csv', sep=';')
    df1['red'] = 1
    df1['white'] = 0
    df2 = pd.read_csv('data/winequality-white.csv', sep=';')
    df2['red'] = 0
    df2['white'] = 1
    wine_quality = pd.concat([df1, df2], axis=0, )
    return traitlets.Bunch(
        data=wine_quality[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                           'pH', 'sulphates', 'alcohol', 'red', 'white']].values,
        target=wine_quality[['quality']].values,
        target_names=wine_quality["quality"].unique(),
        DESCR='red and white wine quality dataset...',
        feature_names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol', 'red', 'white'],
    )


