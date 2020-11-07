import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    csv = "cathrynli.csv"
    df = pd.read_csv(csv)
    df.set_index('filename', inplace=True)

    print(f"Columns: {list(df.columns)}")
    print(df.head())
    plt.title('Skin Pixels to Image Size Ratio against Number of Likes')
    sns.scatterplot(data=df, x="skin2img ratio", y="numOfLikes")
    plt.savefig("scatterplot.png", bbox_inches='tight')
    plt.show()
